"""Parquet storage utilities - simplified for demonstration"""

import polars as pl
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class ParquetStorageManager:
    """
    Manages Parquet file storage and retrieval.
    Simplified version for demonstration purposes - no database dependencies.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for storage (defaults to ./data)
        """
        self.base_path = base_path or Path("./data")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store_market_orders(
        self,
        orders: List[Dict[str, Any]],
        timestamp: datetime,
        filename: Optional[str] = None
    ) -> Path:
        """
        Store market orders in Parquet format.

        Args:
            orders: List of order dictionaries
            timestamp: Timestamp for the snapshot
            filename: Optional custom filename (defaults to timestamped filename)

        Returns:
            Path to the created file
        """
        if not orders:
            raise ValueError("No orders to store")

        # Convert to Polars DataFrame
        df = pl.DataFrame(orders)

        # Add metadata columns
        df = df.with_columns([
            pl.lit(timestamp).alias('fetch_timestamp')
        ])

        # Add region_id if not already present in orders
        if 'region_id' not in df.columns:
            region_id = orders[0].get('region_id', 10000002)  # Default to The Forge
            df = df.with_columns([
                pl.lit(region_id).alias('region_id')
            ])

        # Ensure correct data types
        df = df.with_columns([
            pl.col('order_id').cast(pl.Int64),
            pl.col('type_id').cast(pl.Int32),
            pl.col('location_id').cast(pl.Int64),
            pl.col('system_id').cast(pl.Int32),
            pl.col('region_id').cast(pl.Int32),
            pl.col('volume_total').cast(pl.Int32),
            pl.col('volume_remain').cast(pl.Int32),
            pl.col('min_volume').cast(pl.Int32),
            pl.col('price').cast(pl.Float64),
            pl.col('is_buy_order').cast(pl.Boolean),
            pl.col('duration').cast(pl.Int32),
            pl.col('issued').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%SZ'),
            pl.col('range').cast(pl.Utf8),
            pl.col('fetch_timestamp').cast(pl.Datetime)
        ])

        # Generate filename
        if filename is None:
            filename = f"region_{orders[0].get('region_id', 10000002)}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"

        file_path = self.base_path / filename

        # Write with Snappy compression
        df.write_parquet(
            file_path,
            compression='snappy',
            use_pyarrow=True
        )

        # Verify file was created
        if not file_path.exists():
            raise ValueError(f"Failed to create parquet file: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✅ Stored {len(orders):,} orders to {file_path.name} ({file_size_mb:.2f} MB)")

        return file_path

    def read_parquet(self, file_path: Path) -> pl.DataFrame:
        """
        Read Parquet file.

        Args:
            file_path: Path to the Parquet file

        Returns:
            Polars DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pl.read_parquet(file_path)

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"📖 Read {df.height:,} rows from {file_path.name} ({file_size_mb:.2f} MB)")

        return df

    def read_multiple_parquet(self, file_paths: List[Path]) -> pl.DataFrame:
        """
        Read multiple Parquet files and concatenate.

        Args:
            file_paths: List of paths to Parquet files

        Returns:
            Polars DataFrame with all data concatenated
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        # Read all files
        dfs = []
        for file_path in file_paths:
            if not file_path.exists():
                print(f"⚠️  Skipping missing file: {file_path}")
                continue
            dfs.append(pl.read_parquet(file_path))

        if not dfs:
            raise ValueError("No valid files found")

        # Concatenate all DataFrames
        df = pl.concat(dfs)

        total_size_mb = sum(fp.stat().st_size for fp in file_paths if fp.exists()) / (1024 * 1024)
        print(f"📖 Read {df.height:,} rows from {len(dfs)} files ({total_size_mb:.2f} MB total)")

        return df

    def write_events(
        self,
        events_df,
        filename: str
    ) -> Path:
        """
        Write events DataFrame to Parquet.

        Args:
            events_df: Polars or Pandas DataFrame with events
            filename: Name of the output file

        Returns:
            Path to the created file
        """
        import pandas as pd

        # Check if empty and get count
        if isinstance(events_df, pl.DataFrame):
            if events_df.height == 0:
                raise ValueError("No events to store")
            num_events = events_df.height
        elif isinstance(events_df, pd.DataFrame):
            if len(events_df) == 0:
                raise ValueError("No events to store")
            num_events = len(events_df)
        else:
            raise TypeError("events_df must be a Polars or Pandas DataFrame")

        if not filename.endswith('.parquet'):
            filename = f"{filename}.parquet"

        file_path = self.base_path / filename

        # Write with Snappy compression
        if isinstance(events_df, pl.DataFrame):
            events_df.write_parquet(
                file_path,
                compression='snappy',
                use_pyarrow=True
            )
        else:  # Pandas DataFrame
            events_df.to_parquet(
                file_path,
                compression='snappy',
                engine='pyarrow',
                index=False
            )

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✅ Stored {num_events:,} events to {file_path.name} ({file_size_mb:.2f} MB)")

        return file_path
