"""Add webhook_experiments table

Create Date: 2026-11-23 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6fea3ae46a47"
down_revision = "f4a9b2c7d1e6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "webhook_experiments",
        sa.Column("webhook_id", sa.String(length=256), nullable=False),
        sa.Column("experiment_id", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["webhook_id"],
            ["webhooks.webhook_id"],
            name="fk_webhook_experiments_webhook_id",
            ondelete="cascade",
        ),
        sa.PrimaryKeyConstraint("webhook_id", "experiment_id", name="webhook_experiment_pk"),
    )
    op.create_index(
        "idx_webhook_experiments_experiment_id", "webhook_experiments", ["experiment_id"]
    )


def downgrade():
    op.drop_index("idx_webhook_experiments_experiment_id", "webhook_experiments")
    op.drop_table("webhook_experiments")
