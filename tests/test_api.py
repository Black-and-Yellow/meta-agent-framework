"""
Test Suite — FastAPI Endpoints
===============================
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meta_agent.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_readiness(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data


class TestAgentEndpoints:
    def test_list_agent_types(self):
        response = client.get("/api/v1/agents/types")
        assert response.status_code == 200
        data = response.json()
        assert "agent_types" in data
        types = [t["type"] for t in data["agent_types"]]
        assert "research" in types
        assert "coding" in types

    def test_list_registered(self):
        response = client.get("/api/v1/agents/registered")
        assert response.status_code == 200
        data = response.json()
        assert "registered_types" in data


class TestTaskEndpoints:
    def test_task_not_found(self):
        response = client.get("/api/v1/tasks/nonexistent")
        assert response.status_code == 404

    def test_list_tasks(self):
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestBlueprintEndpoints:
    def test_list_blueprints(self):
        response = client.get("/api/v1/blueprints")
        assert response.status_code == 200

    def test_get_blueprint(self):
        response = client.get("/api/v1/blueprints/test_id")
        assert response.status_code == 200
