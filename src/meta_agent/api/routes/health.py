"""Health check routes — liveness, readiness, and system info."""

from __future__ import annotations

from fastapi import APIRouter

from meta_agent import __version__

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic liveness probe."""
    return {"status": "healthy", "version": __version__}


@router.get("/health/ready")
async def readiness_check() -> dict:
    """Readiness probe — checks that dependencies are available."""
    checks = {
        "api": True,
        "version": __version__,
    }

    # Check Redis
    try:
        from meta_agent.config import get_settings
        settings = get_settings()
        if settings.redis_url:
            import redis
            r = redis.from_url(settings.redis_url, socket_timeout=2)
            r.ping()
            checks["redis"] = True
    except Exception:
        checks["redis"] = False

    # Check ChromaDB
    try:
        import chromadb
        checks["chromadb"] = True
    except ImportError:
        checks["chromadb"] = False

    all_ready = checks.get("api", False)
    return {"ready": all_ready, "checks": checks}
