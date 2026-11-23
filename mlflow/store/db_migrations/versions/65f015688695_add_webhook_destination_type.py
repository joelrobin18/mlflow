"""Add destination_type to webhooks

Create Date: 2026-11-23 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "65f015688695"
down_revision = "6fea3ae46a47"
branch_labels = None
depends_on = None


def upgrade():
    # Every webhook stored before this was a generic one, so the default leaves them as they were.
    with op.batch_alter_table("webhooks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "destination_type",
                sa.String(length=20),
                nullable=False,
                server_default="GENERIC",
            )
        )


def downgrade():
    with op.batch_alter_table("webhooks") as batch_op:
        batch_op.drop_column("destination_type")
