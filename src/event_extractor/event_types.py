"""
Event types and data structures for EVE market event sourcing
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class EventType(Enum):
    """Types of market events we can detect"""
    TRADE = "trade"                    # Volume was traded (partial fill: volume reduced)
    ORDER_OPENED = "order_opened"      # New order appeared
    ORDER_CLOSED = "order_closed"      # Order with volume=0 disappeared (fully filled)
    ORDER_CANCELLED = "order_cancelled" # Order manually cancelled (disappeared, not scheduled to expire)
    ORDER_EXPIRED = "order_expired"    # Order expired naturally (scheduled expiration time reached)
    PRICE_CHANGED = "price_changed"    # Order price was modified


@dataclass
class TradingEvent:
    """
    Represents a single market event with sparse storage.

    This uses a flat structure where different event types use different subsets of fields.
    Parquet's columnar format with RLE compression makes NULL storage nearly free.

    Field usage by event type:
    - Core fields: Always present (event_type, timestamp, type_id, order_id)
    - Delta fields: Used by TRADE, CANCELLED, EXPIRED, PRICE_CHANGED (~85% of events)
    - Full-state fields: Only used by ORDER_OPENED (~15% of events)

    The full-state fields enable self-contained event logs that can reconstruct
    order books without needing anchor snapshots.
    """
    # Core fields (always present)
    event_type: EventType
    timestamp: datetime
    type_id: int
    order_id: Optional[int] = None

    # Delta fields (used in ~85% of events: TRADE, CANCELLED, EXPIRED, PRICE_CHANGED)
    volume: Optional[int] = None           # Volume traded/cancelled/opened
    price: Optional[float] = None          # Order price
    is_buy_order: Optional[bool] = None    # True = buy, False = sell
    system_id: Optional[int] = None        # Solar system ID
    confidence: float = 1.0                # Detection confidence (0.0-1.0)

    # Full-state fields (only ORDER_OPENED - ~15% of events, 85% NULL)
    # These allow reconstructing order books from events alone (no snapshots needed)
    location_id: Optional[int] = None      # Station/structure ID where order is located
    volume_total: Optional[int] = None     # Original total volume when order was placed
    min_volume: Optional[int] = None       # Minimum purchase quantity
    duration: Optional[int] = None         # Order duration in days
    issued: Optional[datetime] = None      # When the order was originally issued
    range: Optional[str] = None            # Order range: 'station', 'region', 'solarsystem', etc.
    region_id: Optional[int] = None        # Region ID (can be derived from system_id)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        # Handle issued field - could be datetime or string from Polars
        issued_str = None
        if self.issued:
            if isinstance(self.issued, datetime):
                issued_str = self.issued.isoformat()
            else:
                issued_str = str(self.issued)  # Already a string from Polars

        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'type_id': self.type_id,
            'order_id': self.order_id,
            'volume': self.volume,
            'price': self.price,
            'is_buy_order': self.is_buy_order,
            'system_id': self.system_id,
            'confidence': self.confidence,
            # Full-state fields (sparse - mostly NULL)
            'location_id': self.location_id,
            'volume_total': self.volume_total,
            'min_volume': self.min_volume,
            'duration': self.duration,
            'issued': issued_str,
            'range': self.range,
            'region_id': self.region_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TradingEvent':
        """Create from dictionary"""
        return cls(
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            type_id=data['type_id'],
            order_id=data.get('order_id'),
            volume=data.get('volume'),
            price=data.get('price'),
            is_buy_order=data.get('is_buy_order'),
            system_id=data.get('system_id'),
            confidence=data.get('confidence', 1.0),
            # Full-state fields
            location_id=data.get('location_id'),
            volume_total=data.get('volume_total'),
            min_volume=data.get('min_volume'),
            duration=data.get('duration'),
            issued=datetime.fromisoformat(data['issued']) if data.get('issued') else None,
            range=data.get('range'),
            region_id=data.get('region_id'),
        )
