"""
Polars-optimized event detector
Leverages: vectorized operations, lazy evaluation, multi-threading
Returns Polars DataFrames for maximum performance (no Python object creation overhead)
"""

import polars as pl
from typing import Set, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .event_types import EventType


class PolarsEventDetector:
    """Event detector optimized for Polars - fully vectorized, multi-threaded, returns DataFrames"""

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def initialize_from_snapshot(
        self,
        snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pl.DataFrame:
        """
        Create initial ORDER_OPENED events from the first snapshot.

        This establishes the baseline state - all existing orders are treated
        as newly opened. This makes the event log self-contained and allows
        reconstructing the order book state from events alone.

        Args:
            snapshot_path: Path to the initial snapshot
            timestamp: Timestamp for these events
            target_type_ids: Optional filter for specific type IDs

        Returns:
            Polars DataFrame with ORDER_OPENED events for all orders in snapshot
        """
        snapshot = pl.read_parquet(snapshot_path)

        # Filter by target type IDs if specified
        if target_type_ids:
            snapshot = snapshot.filter(pl.col('type_id').is_in(list(target_type_ids)))

        print(f"Initializing event log from snapshot: {snapshot.height:,} orders")

        return self._process_new_orders(snapshot, timestamp)

    def detect_events(
        self,
        prev_snapshot_path: Path,
        current_snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pl.DataFrame:
        """
        Detect events between two snapshots using Polars optimizations - returns DataFrame directly.

        Key optimizations:
        - Vectorized operations throughout
        - No Python object creation
        - Multi-threaded joins and aggregations
        - Lazy evaluation where possible

        Returns:
            Polars DataFrame with all detected events
        """
        # Read snapshots
        prev_snapshot = pl.read_parquet(prev_snapshot_path)
        current_snapshot = pl.read_parquet(current_snapshot_path)

        # Filter by target type IDs if specified
        if target_type_ids:
            type_list = list(target_type_ids)
            prev_snapshot = prev_snapshot.filter(pl.col('type_id').is_in(type_list))
            current_snapshot = current_snapshot.filter(pl.col('type_id').is_in(type_list))

        event_dfs = []

        # 1. Detect disappeared orders + partial fills + price changes in one pass
        # Use a single join to avoid multiple scans
        joined = prev_snapshot.join(
            current_snapshot,
            on='order_id',
            how='left',  # Keep all prev orders
            suffix='_curr'
        )

        # Disappeared orders (no match in current)
        disappeared = joined.filter(pl.col('type_id_curr').is_null())
        if disappeared.height > 0:
            event_dfs.append(self._process_disappeared_orders(disappeared, timestamp))

        # Still present orders
        still_present = joined.filter(pl.col('type_id_curr').is_not_null())

        # Partial fills (volume reduced)
        partial_fills = still_present.filter(pl.col('volume_remain') > pl.col('volume_remain_curr'))
        if partial_fills.height > 0:
            event_dfs.append(self._process_partial_fills(partial_fills, timestamp))

        # Price changes
        price_changes = still_present.filter(pl.col('price') != pl.col('price_curr'))
        if price_changes.height > 0:
            event_dfs.append(self._process_price_changes(price_changes, timestamp))

        # 2. Detect new orders (anti-join)
        new_orders = current_snapshot.join(
            prev_snapshot.select(['order_id']),
            on='order_id',
            how='anti'  # Only orders not in prev
        )
        if new_orders.height > 0:
            event_dfs.append(self._process_new_orders(new_orders, timestamp))

        # Concatenate all event DataFrames
        if event_dfs:
            # Use vertical_relaxed to handle potential schema differences, then reorder columns
            combined = pl.concat(event_dfs, how="vertical_relaxed")

            # Ensure consistent column order
            column_order = [
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence',
                'location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range'
            ]
            # Only select columns that exist
            existing_cols = [col for col in column_order if col in combined.columns]
            return combined.select(existing_cols)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(schema={
                'event_type': pl.Utf8,
                'timestamp': pl.Datetime,
                'type_id': pl.Int64,
                'order_id': pl.Int64,
                'volume': pl.Int64,
                'price': pl.Float64,
                'is_buy_order': pl.Boolean,
                'system_id': pl.Int64,
                'confidence': pl.Float64,
                'location_id': pl.Int64,
                'volume_total': pl.Int64,
                'min_volume': pl.Int64,
                'duration': pl.Int64,
                'issued': pl.Utf8,
                'range': pl.Utf8
            })

    def _process_disappeared_orders(
        self,
        disappeared: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """
        Process disappeared orders - vectorized.

        Logic:
        1. If volume_remain = 0 → ORDER_CLOSED (fully filled, order removed from book)
        2. If volume_remain > 0:
           a. If scheduled to expire in interval → ORDER_EXPIRED
           b. If NOT scheduled to expire → ORDER_CANCELLED
        """
        df = disappeared.with_columns([
            # Initialize event type based on volume_remain
            pl.when(pl.col('volume_remain') == 0)
              .then(pl.lit(EventType.ORDER_CLOSED.value))
              .otherwise(pl.lit(EventType.ORDER_CANCELLED.value))
              .alias('event_type'),

            # Initialize confidence
            pl.when(pl.col('volume_remain') == 0)
              .then(pl.lit(1.0))
              .otherwise(pl.lit(0.90))
              .alias('confidence'),

            pl.lit(timestamp).alias('timestamp'),
            pl.col('volume_remain').alias('volume')
        ])

        # Check for expired orders (requires iteration for complex date logic)
        # Convert to pandas for iteration, then back to polars
        if 'issued' in df.columns and 'duration' in df.columns:
            df_pandas = df.to_pandas()
            for idx, row in df_pandas[df_pandas['volume_remain'] > 0].iterrows():
                if row['issued'] is not None and row['duration'] is not None:
                    try:
                        issued = row['issued']
                        if isinstance(issued, str):
                            from dateutil import parser
                            issued = parser.parse(issued)
                        duration_days = int(row['duration'])
                        expiration_time = issued + timedelta(days=duration_days)
                        time_to_expiration = (expiration_time - timestamp).total_seconds()

                        # Assuming 5-min snapshots
                        if -300 <= time_to_expiration <= 0:
                            df_pandas.at[idx, 'event_type'] = EventType.ORDER_EXPIRED.value
                            df_pandas.at[idx, 'confidence'] = 0.95
                    except (ValueError, TypeError):
                        pass
            df = pl.from_pandas(df_pandas)

        # Select columns for output - include full-state fields as NULL
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]

        # Add NULL columns for full-state fields (sparse schema)
        full_state_fields = ['location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range']
        result = df.select(columns)
        for field in full_state_fields:
            result = result.with_columns(pl.lit(None).alias(field))

        return result

    def _process_partial_fills(
        self,
        partial_fills: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """Process partial fills - fully vectorized"""
        df = partial_fills.with_columns([
            pl.lit(EventType.TRADE.value).alias('event_type'),
            pl.lit(timestamp).alias('timestamp'),
            (pl.col('volume_remain') - pl.col('volume_remain_curr')).alias('volume'),
            pl.lit(1.0).alias('confidence')
        ])

        # Select columns for output
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]

        # Add NULL columns for full-state fields (sparse schema)
        full_state_fields = ['location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range']
        result = df.select(columns)
        for field in full_state_fields:
            result = result.with_columns(pl.lit(None).alias(field))

        return result

    def _process_price_changes(
        self,
        price_changes: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """Process price changes - fully vectorized"""
        df = price_changes.with_columns([
            pl.lit(EventType.PRICE_CHANGED.value).alias('event_type'),
            pl.lit(timestamp).alias('timestamp'),
            pl.col('volume_remain_curr').alias('volume'),
            pl.col('price_curr').alias('price'),
            pl.lit(1.0).alias('confidence')
        ])

        # Select columns for output
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]

        # Add NULL columns for full-state fields (sparse schema)
        full_state_fields = ['location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range']
        result = df.select(columns)
        for field in full_state_fields:
            result = result.with_columns(pl.lit(None).alias(field))

        return result

    def _process_new_orders(
        self,
        new_orders: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """Process new orders - capture full state, fully vectorized"""
        # Add event metadata
        df = new_orders.with_columns([
            pl.lit(EventType.ORDER_OPENED.value).alias('event_type'),
            pl.lit(timestamp).alias('timestamp'),
            pl.col('volume_remain').alias('volume'),
            pl.lit(1.0).alias('confidence')
        ])

        # Select base columns
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]
        result = df.select(columns)

        # Add full-state fields if they exist, otherwise NULL
        full_state_fields = ['location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range']
        for field in full_state_fields:
            if field in df.columns:
                result = result.with_columns(df[field].alias(field))
            else:
                result = result.with_columns(pl.lit(None).alias(field))

        return result
