"""Aggregate router for API v1. Feature routers are included here as they land."""

from fastapi import APIRouter

from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.papers import router as papers_router
from app.api.v1.endpoints.search import router as search_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(papers_router)
api_router.include_router(search_router)
