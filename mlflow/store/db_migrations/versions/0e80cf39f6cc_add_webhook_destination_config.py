"""Add destination_config to webhooks

Create Date: 2026-11-23 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0e80cf39f6cc"
down_revision = "65f015688695"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("webhooks") as batch_op:
        batch_op.add_column(sa.Column("destination_config", sa.String(length=1000), nullable=True))


def downgrade():
    with op.batch_alter_table("webhooks") as batch_op:
        batch_op.drop_column("destination_config")
