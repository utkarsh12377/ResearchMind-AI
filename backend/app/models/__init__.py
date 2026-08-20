"""Import all ORM models here so Alembic autogenerate can discover them via Base.metadata."""

from app.models.api_key import ApiKey
from app.models.paper import Paper, PaperStatus
from app.models.paper_asset import AssetKind, PaperAsset
from app.models.paper_reference import PaperReference
from app.models.user import User
from app.models.workspace import Workspace

__all__ = [
    "ApiKey",
    "AssetKind",
    "Paper",
    "PaperAsset",
    "PaperReference",
    "PaperStatus",
    "User",
    "Workspace",
]
