"""SQLAlchemy ORM models for STRIVE backend tables."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, BigInteger, SmallInteger, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """Base declarative model for ORM mappings."""


class RoadSegment(Base):
    """Road network edge enriched with routing and historical risk metadata."""

    __tablename__ = "road_segments"

    segment_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    u: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    v: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    geometry: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    road_class: Mapped[str] = mapped_column(String(64), nullable=False, default="unclassified")
    speed_limit_kmh: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    length_m: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    historical_accident_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Accident(Base):
    """Historical crash record snapped to a road segment."""

    __tablename__ = "accidents"

    accident_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("road_segments.segment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    severity: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
