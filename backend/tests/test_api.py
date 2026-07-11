import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from main import app

def _fake_openai_response(data: dict):
    """Build a stand-in for an OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))]
    )

async def test_create_ticket_endpoint(monkeypatch):
    """Test POST /tickets endpoint with mocked OpenAI calls and mocked DB write"""
    monkeypatch.setattr(
        "agents.triage_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({
            "category": "hardware",
            "priority": "high",
            "business_impact": "high"
        }))
    )
    monkeypatch.setattr(
        "agents.troubleshooting_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({
            "steps": [
                "Check the power cable and outlet",
                "Hold the power button for 30 seconds",
                "Try booting into safe mode"
            ],
            "estimated_effort": "easy"
        }))
    )
    monkeypatch.setattr(
        "agents.resolution_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({
            "message": "We're on it - please try the steps below and let us know if the issue persists.",
            "tone": "reassuring"
        }))
    )
    monkeypatch.setattr(
        "main.tickets_collection.insert_one",
        AsyncMock(return_value=SimpleNamespace(inserted_id="fake_id"))
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/tickets",
            json={
                "issue": "laptop won't boot",
                "system": "Dell Latitude laptop",
                "impact": "can't start new job at HP"
                }
        )
    assert response.status_code == 200
    data = response.json()
    assert "triage" in data
    assert "troubleshooting" in data
    assert "resolution" in data