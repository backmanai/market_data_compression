"""
Dask-optimized event detector
Leverages: parallel processing, lazy evaluation, multi-core computation
Returns DataFrames for maximum performance (no Python object creation overhead)
Uses Pandas backend for single-pair processing
"""

import dask
import dask.dataframe as dd
import pandas as pd
from typing import Set, Optional, List
from datetime import datetime
from pathlib import Path

from .event_detector_pandas import PandasEventDetector


class DaskEventDetector:
    """
    Event detector optimized for Dask - parallel processing across CPU cores

    Key optimization: Process multiple snapshot pairs in parallel
    Uses Pandas detector for single-pair logic (avoiding code duplication)
    """

    def __init__(self, confidence_threshold: float = 0.8, n_workers: int = 4):
        self.confidence_threshold = confidence_threshold
        self.n_workers = n_workers
        # Use Pandas detector for core logic
        self.pandas_detector = PandasEventDetector(confidence_threshold)

    def initialize_from_snapshot(
        self,
        snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Create initial ORDER_OPENED events from the first snapshot.

        Uses Dask for parallel I/O, then Pandas for processing.

        Args:
            snapshot_path: Path to the initial snapshot
            timestamp: Timestamp for these events
            target_type_ids: Optional filter for specific type IDs

        Returns:
            Pandas DataFrame with ORDER_OPENED events for all orders in snapshot
        """
        # Dask parallel I/O, then delegate to Pandas
        # (For single file initialization, just use Pandas directly)
        return self.pandas_detector.initialize_from_snapshot(
            snapshot_path, timestamp, target_type_ids
        )

    def detect_events_batch(
        self,
        snapshot_pairs: List[tuple],
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Process multiple snapshot pairs in PARALLEL using Dask delayed

        This is where Dask shines - processing many files concurrently

        Args:
            snapshot_pairs: List of (prev_path, curr_path, timestamp) tuples
            target_type_ids: Optional filter for specific type IDs

        Returns:
            Pandas DataFrame with all events from all snapshot pairs
        """
        # Create delayed tasks for parallel execution
        tasks = [
            dask.delayed(self._detect_events_single)(prev, curr, ts, target_type_ids)
            for prev, curr, ts in snapshot_pairs
        ]

        # Compute all tasks in parallel across CPU cores
        results = dask.compute(*tasks, num_workers=self.n_workers)

        # Concatenate all DataFrames
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])

    def detect_events(
        self,
        prev_snapshot_path: Path,
        current_snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Single snapshot pair processing - delegates to Pandas backend

        For single pairs, Dask has overhead without benefit,
        so we just use Pandas directly.

        Returns:
            Pandas DataFrame with detected events
        """
        return self._detect_events_single(
            prev_snapshot_path,
            current_snapshot_path,
            timestamp,
            target_type_ids
        )

    def _detect_events_single(
        self,
        prev_snapshot_path: Path,
        current_snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Process a single snapshot pair - delegates to Pandas detector

        This avoids code duplication and ensures consistency across implementations.
        """
        return self.pandas_detector.detect_events(
            prev_snapshot_path,
            current_snapshot_path,
            timestamp,
            target_type_ids
        )

    def detect_events_parallel_by_type(
        self,
        prev_snapshot_path: Path,
        current_snapshot_path: Path,
        timestamp: datetime,
        target_type_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Parallel processing of a single snapshot pair by partitioning by type_id

        Useful for very large single files (> 1 GB)

        Args:
            prev_snapshot_path: Path to previous snapshot
            current_snapshot_path: Path to current snapshot
            timestamp: Event timestamp
            target_type_ids: Optional filter for specific type IDs

        Returns:
            Pandas DataFrame with all detected events
        """
        # Read snapshots
        prev_snapshot = pd.read_parquet(prev_snapshot_path)
        current_snapshot = pd.read_parquet(current_snapshot_path)

        # Filter by target type IDs if specified
        if target_type_ids:
            prev_snapshot = prev_snapshot[prev_snapshot['type_id'].isin(target_type_ids)]
            current_snapshot = current_snapshot[current_snapshot['type_id'].isin(target_type_ids)]

        # Get unique type_ids
        all_type_ids = set(prev_snapshot['type_id'].unique()) | set(current_snapshot['type_id'].unique())

        if len(all_type_ids) == 0:
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])

        # Partition type_ids into chunks for parallel processing
        num_partitions = min(self.n_workers, max(1, len(all_type_ids) // 1000))
        type_ids_list = sorted(all_type_ids)
        partition_size = len(type_ids_list) // num_partitions + 1

        type_partitions = [
            set(type_ids_list[i:i + partition_size])
            for i in range(0, len(type_ids_list), partition_size)
        ]

        # Create delayed tasks for each partition
        tasks = [
            dask.delayed(self._process_type_partition)(
                prev_snapshot, current_snapshot, timestamp, partition
            )
            for partition in type_partitions
        ]

        # Execute in parallel
        results = dask.compute(*tasks, num_workers=self.n_workers)

        # Concatenate results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])

    def _process_type_partition(
        self,
        prev_snapshot: pd.DataFrame,
        current_snapshot: pd.DataFrame,
        timestamp: datetime,
        type_ids: Set[int]
    ) -> pd.DataFrame:
        """
        Process a partition of type_ids

        Args:
            prev_snapshot: Previous snapshot (full)
            current_snapshot: Current snapshot (full)
            timestamp: Event timestamp
            type_ids: Set of type_ids to process in this partition

        Returns:
            DataFrame with events for this partition
        """
        # Filter to just this partition's type_ids
        prev_part = prev_snapshot[prev_snapshot['type_id'].isin(type_ids)]
        curr_part = current_snapshot[current_snapshot['type_id'].isin(type_ids)]

        # Process using Pandas detector
        # We'll use the core merge logic directly
        if len(prev_part) == 0 and len(curr_part) == 0:
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])

        event_dfs = []

        # Merge
        merged = prev_part.merge(
            curr_part,
            on='order_id',
            how='outer',
            indicator=True,
            suffixes=('', '_curr')
        )

        # Process each event type
        disappeared = merged[merged['_merge'] == 'left_only']
        if len(disappeared) > 0:
            event_dfs.append(self.pandas_detector._process_disappeared_orders(disappeared, timestamp))

        both = merged[merged['_merge'] == 'both']
        partial_fills = both[both['volume_remain'] > both['volume_remain_curr']]
        if len(partial_fills) > 0:
            event_dfs.append(self.pandas_detector._process_partial_fills(partial_fills, timestamp))

        price_changes = both[both['price'] != both['price_curr']]
        if len(price_changes) > 0:
            event_dfs.append(self.pandas_detector._process_price_changes(price_changes, timestamp))

        new_orders = merged[merged['_merge'] == 'right_only']
        if len(new_orders) > 0:
            event_dfs.append(self.pandas_detector._process_new_orders(new_orders, timestamp))

        # Concatenate partition results
        if event_dfs:
            return pd.concat(event_dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=[
                'event_type', 'timestamp', 'type_id', 'order_id',
                'volume', 'price', 'is_buy_order', 'system_id', 'confidence'
            ])
