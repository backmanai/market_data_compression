#!/usr/bin/env python3
"""
Fetch EVE Online market data from ESI API and convert to Parquet format.

This script:
1. Fetches market orders from the EVE Online ESI API
2. Saves the raw data as JSON
3. Converts the JSON data to Parquet format with Snappy compression

Usage:
    python fetch_and_convert.py [--region REGION_ID] [--output-dir OUTPUT_DIR]
"""

import json
import sys
from pathlib import Path
import argparse
import polars as pl

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.eve_api import EVEAPIClient


def main():
    parser = argparse.ArgumentParser(
        description="Fetch EVE Online market data and convert to Parquet format"
    )
    parser.add_argument(
        "--region",
        type=int,
        default=10000002,
        help="EVE Online region ID (default: 10000002 = The Forge)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/demo"),
        help="Output directory for JSON and Parquet files (default: ./data/demo)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVE Online Market Data Fetcher")
    print("=" * 70)
    print(f"Region ID: {args.region}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Initialize API client
    api_client = EVEAPIClient()

    # Fetch market orders
    print("📡 Fetching market data from ESI API...")
    print("⏱️  This will take ~60 seconds due to pagination and rate limiting")
    print()

    result = api_client.get_market_orders(args.region)
    orders = result['orders']
    fetch_time = result['fetch_duration']

    print()
    print(f"✅ Fetched {len(orders):,} market orders")
    print(f"⏱️  Fetch duration: {fetch_time:.1f} seconds")
    print(f"📄 Pages fetched: {result['pages_fetched']}")
    print()

    # Save as JSON
    json_path = args.output_dir / "market_orders.json"

    print(f"💾 Saving raw JSON to: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(orders, f, indent=2)

    json_size_bytes = json_path.stat().st_size
    json_size_mb = json_size_bytes / (1024 * 1024)
    print(f"   File size: {json_size_mb:.2f} MB ({json_size_bytes:,} bytes)")
    print(f"   Avg bytes per order: {json_size_bytes / len(orders):.1f} bytes")
    print()

    # Convert to Parquet
    parquet_path = args.output_dir / "market_orders.parquet"

    print(f"🔄 Converting to Parquet format...")

    # Convert to Polars DataFrame
    df = pl.DataFrame(orders)

    # Write with Snappy compression (no timestamp metadata)
    df.write_parquet(parquet_path, compression='snappy')

    print(f"✅ Stored {df.height:,} orders to {parquet_path.name}")

    parquet_size_bytes = parquet_path.stat().st_size
    parquet_size_mb = parquet_size_bytes / (1024 * 1024)
    compression_ratio = json_size_bytes / parquet_size_bytes

    print()
    print("=" * 70)
    print("📊 Compression Results")
    print("=" * 70)
    print(f"Raw JSON:        {json_size_mb:8.2f} MB ({json_path.name})")
    print(f"Parquet:         {parquet_size_mb:8.2f} MB ({parquet_path.name})")
    print(f"Compression:     {compression_ratio:8.1f}× reduction")
    print(f"Avg per order:   {parquet_size_bytes / len(orders):8.1f} bytes (Parquet)")
    print(f"                 vs {json_size_bytes / len(orders):.1f} bytes (JSON)")
    print("=" * 70)
    print()
    print("✅ Done! Files saved to:")
    print(f"   📄 JSON:    {json_path}")
    print(f"   📦 Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
