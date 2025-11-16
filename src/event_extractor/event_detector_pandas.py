"""
Pandas-optimized event detector
Leverages: vectorized operations, efficient indexing, optimizations
Returns DataFrames for maximum performance (no Python object creation overhead)
"""

import pandas as pd
import numpy as np
from typing import Set, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .event_types import EventType


class PandasEventDetector:
    """Event detector optimized for Pandas - fully vectorized, returns DataFrames"""

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def initialize_from_snapshot(
        self,
        snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
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
            DataFrame with ORDER_OPENED events for all orders in snapshot
        """
        snapshot = pd.read_parquet(snapshot_path)

        # Filter by target type IDs if specified
        if target_type_ids:
            snapshot = snapshot[snapshot['type_id'].isin(target_type_ids)]

        print(f"🌱 Initializing event log from snapshot: {len(snapshot):,} orders")

        # Add event metadata columns
        events_df = snapshot.copy()
        events_df['event_type'] = EventType.ORDER_OPENED.value
        events_df['timestamp'] = timestamp
        events_df['volume'] = events_df['volume_remain']
        events_df['confidence'] = 1.0

        # Select and rename columns to match event schema
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence',
            # Full-state fields
            'location_id', 'volume_total', 'min_volume', 'duration',
            'issued', 'range', 'region_id'
        ]

        # Only include columns that exist
        available_columns = [col for col in columns if col in events_df.columns]
        return events_df[available_columns].reset_index(drop=True)

    def detect_events(
        self,
        prev_snapshot_path: Path,
        current_snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Detect events using Pandas optimizations - returns DataFrame directly.

        Key optimizations:
        - Vectorized operations throughout
        - No Python object creation
        - Direct DataFrame manipulation
        - Efficient merge operations

        Returns:
            DataFrame with all detected events
        """
        # Read snapshots
        prev_snapshot = pd.read_parquet(prev_snapshot_path)
        current_snapshot = pd.read_parquet(current_snapshot_path)

        # Filter by target type IDs if specified
        if target_type_ids:
            prev_snapshot = prev_snapshot[prev_snapshot['type_id'].isin(target_type_ids)]
            current_snapshot = current_snapshot[current_snapshot['type_id'].isin(target_type_ids)]

        # Single merge operation to avoid multiple scans
        merged = prev_snapshot.merge(
            current_snapshot,
            on='order_id',
            how='outer',
            indicator=True,
            suffixes=('', '_curr')
        )

        event_dfs = []

        # 1. Disappeared orders (left_only)
        disappeared = merged[merged['_merge'] == 'left_only']
        if len(disappeared) > 0:
            event_dfs.append(self._process_disappeared_orders(disappeared, timestamp))

        # 2. Orders in both snapshots
        both = merged[merged['_merge'] == 'both']

        # Partial fills (volume reduced) - vectorized comparison
        partial_fills = both[both['volume_remain'] > both['volume_remain_curr']]
        if len(partial_fills) > 0:
            event_dfs.append(self._process_partial_fills(partial_fills, timestamp))

        # Price changes - vectorized comparison
        price_changes = both[both['price'] != both['price_curr']]
        if len(price_changes) > 0:
            event_dfs.append(self._process_price_changes(price_changes, timestamp))

        # 3. New orders (right_only)
        new_orders = merged[merged['_merge'] == 'right_only']
        if len(new_orders) > 0:
            event_dfs.append(self._process_new_orders(new_orders, timestamp))

        # Concatenate all event DataFrames
        if event_dfs:
            return pd.concat(event_dfs, ignore_index=True)
        else:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])

    def _process_disappeared_orders(
        self,
        disappeared: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Process disappeared orders - vectorized.

        Logic:
        1. If volume_remain = 0 → ORDER_CLOSED (fully filled, order removed from book)
        2. If volume_remain > 0:
           a. If scheduled to expire in interval → ORDER_EXPIRED
           b. If NOT scheduled to expire → ORDER_CANCELLED
        """
        df = disappeared.copy()

        # Initialize event type based on volume_remain
        df['event_type'] = np.where(
            df['volume_remain'] == 0,
            EventType.ORDER_CLOSED.value,
            EventType.ORDER_CANCELLED.value  # Default, will refine below
        )

        # Initialize confidence
        df['confidence'] = np.where(
            df['volume_remain'] == 0,
            1.0,  # ORDER_CLOSED
            0.90  # ORDER_CANCELLED (default)
        )

        # Check for expired orders (more complex, requires iteration)
        if 'issued' in df.columns and 'duration' in df.columns:
            for idx, row in df[df['volume_remain'] > 0].iterrows():
                if pd.notna(row['issued']) and pd.notna(row['duration']):
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
                            df.at[idx, 'event_type'] = EventType.ORDER_EXPIRED.value
                            df.at[idx, 'confidence'] = 0.95
                    except (ValueError, TypeError):
                        pass

        df['timestamp'] = timestamp
        df['volume'] = df['volume_remain']

        # Select columns for output
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]
        return df[columns]

    def _process_partial_fills(
        self,
        partial_fills: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Process partial fills - fully vectorized"""
        df = partial_fills.copy()

        df['event_type'] = EventType.TRADE.value
        df['timestamp'] = timestamp
        df['volume'] = df['volume_remain'] - df['volume_remain_curr']  # Volume traded
        df['confidence'] = 1.0

        # Select columns for output
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]
        return df[columns]

    def _process_price_changes(
        self,
        price_changes: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Process price changes - fully vectorized"""
        df = price_changes.copy()

        df['event_type'] = EventType.PRICE_CHANGED.value
        df['timestamp'] = timestamp
        df['volume'] = df['volume_remain_curr']  # Current volume
        df['price'] = df['price_curr']  # New price
        df['confidence'] = 1.0

        # Select columns for output
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]
        return df[columns]

    def _process_new_orders(
        self,
        new_orders: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Process new orders - capture full state, fully vectorized"""
        df = new_orders.copy()

        df['event_type'] = EventType.ORDER_OPENED.value
        df['timestamp'] = timestamp
        df['confidence'] = 1.0

        # Map _curr columns to standard names
        df['type_id'] = df['type_id_curr']
        df['volume'] = df['volume_remain_curr']
        df['price'] = df['price_curr']
        df['is_buy_order'] = df['is_buy_order_curr']
        df['system_id'] = df['system_id_curr']

        # Select columns including full-state fields
        columns = [
            'event_type', 'timestamp', 'type_id', 'order_id',
            'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
        ]

        # Add full-state fields if they exist
        full_state_fields = ['location_id', 'volume_total', 'min_volume', 'duration', 'issued', 'range', 'region_id']
        for field in full_state_fields:
            curr_field = f'{field}_curr'
            if curr_field in df.columns:
                df[field] = df[curr_field]
                columns.append(field)

        return df[columns]
