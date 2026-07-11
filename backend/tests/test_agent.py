import pytest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from models.schemas import TicketRequest
from agents.triage_agent import triage_ticket, TriageResult

def _fake_openai_response(data: dict):
    """Build a stand-in for an OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))]
    )

def test_ticket_request_schema():
    """Test TicketRequest schema validation"""
    ticket = TicketRequest(
        issue = "laptop won't boot",
        system = "Dell Latitude laptop",
        impact = "blocked from starting new job at HP"
    )
    assert ticket.issue == "laptop won't boot"
    assert ticket.system == "Dell Latitude laptop"
    assert ticket.impact == "blocked from starting new job at HP"

def test_triage_result_schema():
    """Test TriageResult schema validation"""
    result = TriageResult(
        category = "hardware",
        priority = "high",
        business_impact = "high"
    )
    assert result.category == "hardware"
    assert result.priority == "high"
    assert result.business_impact == "high"

@pytest.mark.asyncio
async def test_triage_ticket(monkeypatch):
    """Test triage agent against a mocked OpenAI response (no real API call)"""
    ticket = TicketRequest(
        issue = "laptop won't boot",
        system = "Dell Latitude laptop",
        impact = "can't start new job"
    )

    mock_create = AsyncMock(return_value=_fake_openai_response({
        "category": "hardware",
        "priority": "high",
        "business_impact": "high"
    }))
    monkeypatch.setattr("agents.triage_agent.client.chat.completions.create", mock_create)

    result = await triage_ticket(ticket)

    mock_create.assert_awaited_once()
    assert result.category in ["hardware", "software", "network", "account", "other"]
    assert result.priority in ["low", "medium", "high"]
    assert result.business_impact in ["low", "medium", "high"]