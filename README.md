# Market Data Compression: From JSON to Delta Events

**CAS Machine Intelligence - Module D: Advanced Big Data**
**ZHAW School of Engineering**
**Hand-in Project**

---

## Overview

This project demonstrates a highly efficient data compression pipeline for handling massive market data streams from the Eve Online in-game economy. By combining columnar storage (Parquet) with event sourcing, we achieve a **2,142× compression ratio** compared to raw JSON while maintaining complete data fidelity and analytical value.

### Problem Statement

Eve Online features a player-driven economy with ~30,000 concurrent players. The most active trading region ("The Forge") contains approximately 400,000 open market orders at any given time. Market data refreshes every 5 minutes via public REST endpoints.

**Challenge:**
- Raw JSON snapshot: **121 MB**
- Daily snapshots (288 @ 5-min intervals): **34.8 GB/day**
- This volume is unsustainable for local storage and prohibitively expensive for cloud storage

### Solution

A two-stage compression pipeline:

1. **Stage 1: JSON → Parquet**
   Columnar storage with Snappy compression: **12.7× reduction**

2. **Stage 2: Event Sourcing**
   Store only changes (deltas) between snapshots: **150.9× reduction** over naive Parquet storage

**Final Result:** 34.8 GB/day → **16 MB/day** (2,142× compression)

---

## Architecture

### Stage 1: Columnar Compression

Conversion from JSON to Parquet with Snappy compression leverages:

- **Columnar Layout:** Data stored by column, enabling superior compression for repeated values
- **Dictionary Encoding:** Replaces repeated values (type IDs, locations) with integer references
- **Binary Encoding:** Efficient storage of numbers, timestamps, and booleans
- **Snappy Compression:** Fast general-purpose compression optimized for analytics workloads

**Results:**
- 121 MB → 9.5 MB per snapshot
- 315.2 bytes → 24.8 bytes per order

### Stage 2: Event Sourcing

Instead of storing full snapshots, we extract six types of market events:

```python
TRADE              # Volume traded (partial fill)
ORDER_OPENED       # New order appeared
ORDER_CLOSED       # Order fully filled (volume=0)
ORDER_CANCELLED    # Manual cancellation
ORDER_EXPIRED      # Natural expiration
PRICE_CHANGED      # Price modification
```

**Two-Phase Process:**

1. **Initialization:** Create baseline from first snapshot (1 ORDER_OPENED per order)
2. **Delta Extraction:** Extract only changes between consecutive snapshots

**Key Insight:** Only **0.27%** of orders change per 5-minute interval

**Results:**
- Daily events: 375,338 (initialization) + 293,601 (deltas) = 668,939 events
- Storage: 16.26 MB/day vs. 2,455 MB/day for naive Parquet snapshots
- Full order book can be reconstructed at any point in time

---

## Compression Results

| Storage Method | Size/Day | Compression Ratio |
|---------------|----------|-------------------|
| Raw JSON | 34,827 MB | 1.0× (baseline) |
| Parquet Snapshots | 2,455 MB | 14.2× |
| Event Sourcing (Parquet) | **16 MB** | **2,142×** |

---

## Project Structure

```
market_data_compression/
├── README.md                                      # This file
├── market_data_compression.ipynb                  # Main demonstration notebook
├── data/
│   ├── demo/                                      # Generated demo files
│   │   ├── market_orders.json                     # Raw API response
│   │   ├── market_orders.parquet                  # Stage 1 output
│   │   └── demo_events.parquet                    # Stage 2 output
│   └── snapshots/                                 # Sample consecutive snapshots
│       ├── region_10000002_2025-10-22T07-00-00+00-00.parquet
│       └── region_10000002_2025-10-22T07-05-00+00-00.parquet
└── src/
    ├── event_extractor/
    │   ├── event_detector_polars.py               # Core event detection logic
    │   └── event_types.py                         # Event type definitions
    └── utils/
        ├── eve_api.py                             # Eve Online API client
        ├── fetch_and_convert.py                   # Data fetching script
        └── parquet_storage.py                     # Parquet storage utilities
```

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
# Install dependencies
pip install polars requests
```

### Running the Demonstration

1. **Fetch Fresh Market Data:**

```bash
python src/utils/fetch_and_convert.py
```

This script fetches ~400k market orders from the Eve Online ESI API (~60 seconds) and generates both JSON and Parquet files in `./data/demo/`.

2. **Run Main Notebook:**

Open and execute [market_data_compression.ipynb](market_data_compression.ipynb) to see:
- Stage 1: JSON → Parquet compression analysis
- Stage 2: Event sourcing demonstration with real data
- Daily storage projections
- Complete compression summary

---

## Key Technologies

- **Polars:** High-performance DataFrame library (Rust-based, faster than Pandas)
- **Parquet:** Columnar storage format with built-in compression
- **Snappy Compression:** Fast compression algorithm optimized for analytics
- **Event Sourcing:** Change data capture pattern for efficient storage

---

## Analytical Value

Beyond compression, this approach provides rich analytical insights:

1. **Market Dynamics:**
   - Trade events reveal executed prices and volumes (not publicly available)
   - Price change events show trader competition patterns
   - Order lifecycle tracking (open → partial fills → closed)

2. **Point-in-Time Reconstruction:**
   - Full order book can be reconstructed at any historical timestamp
   - Enables backtesting of trading strategies

3. **Scalability:**
   - 16 MB/day enables years of history on local storage
   - Low data volumes reduce cloud storage costs by 99.9%

---

## Author

**Fredrik Backman (Student)**

**ZHAW CAS Machine Intelligence**
Module D: Advanced Big Data
2025

---

## License

This project is submitted as part of the CAS Machine Intelligence program at ZHAW and is intended for educational purposes.

---

## References

- [Eve Online ESI API](https://esi.evetech.net/ui/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [Apache Parquet Format](https://parquet.apache.org/)
- [Event Sourcing Pattern](https://martinfowler.com/eaaDev/EventSourcing.html)
