"""add extracted entities, experiment results, and paper publication metadata

Revision ID: 3c81f0a25d47
Revises: 069eb7bec6fe
Create Date: 2026-08-21 09:42:18.114522

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c81f0a25d47'
down_revision: Union[str, Sequence[str], None] = '069eb7bec6fe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('papers', sa.Column('published_year', sa.Integer(), nullable=True))
    op.add_column('papers', sa.Column('venue', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_papers_published_year'), 'papers', ['published_year'], unique=False)

    op.create_table(
        'extracted_entities',
        sa.Column('paper_id', sa.Uuid(), nullable=False),
        sa.Column('label', sa.String(length=32), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_entity_label_key', 'extracted_entities', ['label', 'key'], unique=False)
    op.create_index(
        'uq_entity_paper_label_key',
        'extracted_entities',
        ['paper_id', 'label', 'key'],
        unique=True,
    )

    op.create_table(
        'experiment_results',
        sa.Column('paper_id', sa.Uuid(), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('model_key', sa.String(length=255), nullable=False),
        sa.Column('dataset_name', sa.String(length=255), nullable=True),
        sa.Column('dataset_key', sa.String(length=255), nullable=True),
        sa.Column('task_name', sa.String(length=255), nullable=True),
        sa.Column('metric_name', sa.String(length=128), nullable=False),
        sa.Column('metric_key', sa.String(length=128), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(length=32), nullable=True),
        sa.Column('split', sa.String(length=64), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_experiment_lookup', 'experiment_results', ['dataset_key', 'metric_key'], unique=False)
    op.create_index('ix_experiment_paper', 'experiment_results', ['paper_id'], unique=False)
    op.create_index(
        op.f('ix_experiment_results_model_key'), 'experiment_results', ['model_key'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_experiment_results_model_key'), table_name='experiment_results')
    op.drop_index('ix_experiment_paper', table_name='experiment_results')
    op.drop_index('ix_experiment_lookup', table_name='experiment_results')
    op.drop_table('experiment_results')

    op.drop_index('uq_entity_paper_label_key', table_name='extracted_entities')
    op.drop_index('ix_entity_label_key', table_name='extracted_entities')
    op.drop_table('extracted_entities')

    op.drop_index(op.f('ix_papers_published_year'), table_name='papers')
    op.drop_column('papers', 'venue')
    op.drop_column('papers', 'published_year')
