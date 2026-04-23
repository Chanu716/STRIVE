"""Initial road segment and accident schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260420_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    geometry_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

    op.create_table(
        "road_segments",
        sa.Column("segment_id", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column("u", sa.BigInteger(), nullable=False),
        sa.Column("v", sa.BigInteger(), nullable=False),
        sa.Column("geometry", geometry_type, nullable=False),
        sa.Column("road_class", sa.String(length=64), nullable=False),
        sa.Column("speed_limit_kmh", sa.Float(), nullable=False, server_default="50"),
        sa.Column("length_m", sa.Float(), nullable=False, server_default="0"),
        sa.Column("historical_accident_rate", sa.Float(), nullable=False, server_default="0"),
    )
    op.create_index("ix_road_segments_u", "road_segments", ["u"])
    op.create_index("ix_road_segments_v", "road_segments", ["v"])

    op.create_table(
        "accidents",
        sa.Column("accident_id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("segment_id", sa.String(length=255), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("severity", sa.SmallInteger(), nullable=False, server_default="3"),
        sa.ForeignKeyConstraint(["segment_id"], ["road_segments.segment_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_accidents_segment_id", "accidents", ["segment_id"])


def downgrade() -> None:
    op.drop_index("ix_accidents_segment_id", table_name="accidents")
    op.drop_table("accidents")
    op.drop_index("ix_road_segments_v", table_name="road_segments")
    op.drop_index("ix_road_segments_u", table_name="road_segments")
    op.drop_table("road_segments")
