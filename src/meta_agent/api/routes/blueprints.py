"""Blueprint routes — view and inspect generated blueprints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/blueprints")
async def list_blueprints() -> dict:
    """List recently generated blueprints."""
    return {"blueprints": [], "note": "Blueprint storage integration point"}


@router.get("/blueprints/{blueprint_id}")
async def get_blueprint(blueprint_id: str) -> dict:
    """Get a specific blueprint by ID."""
    return {
        "blueprint_id": blueprint_id,
        "note": "Blueprint retrieval integration point",
    }
