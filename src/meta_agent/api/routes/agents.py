"""Agent routes — list available agent types and their descriptions."""

from __future__ import annotations

from fastapi import APIRouter

from meta_agent.agents.factory import AgentFactory
from meta_agent.schemas.blueprint import AgentType

router = APIRouter()


@router.get("/agents/types")
async def list_agent_types() -> dict:
    """List all available agent types."""
    types = [
        {
            "type": at.value,
            "name": at.value.replace("_", " ").title(),
        }
        for at in AgentType
    ]
    return {"agent_types": types}


@router.get("/agents/registered")
async def list_registered_agents() -> dict:
    """List agent types registered in the factory."""
    factory = AgentFactory()
    return {"registered_types": factory.list_types()}
