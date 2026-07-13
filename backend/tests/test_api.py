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


async def _collect_sse_events(response):
    """Parse an SSE response body into a list of (event_type, data) tuples."""
    events = []
    event_type = None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data = json.loads(line[len("data:"):].strip())
            events.append((event_type, data))
            event_type = None
    return events


def _mock_success_pipeline(monkeypatch, triage_data):
    monkeypatch.setattr(
        "agents.triage_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(triage_data)),
    )
    monkeypatch.setattr(
        "agents.troubleshooting_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({
            "steps": ["Restart the app", "Clear the cache", "Reinstall if needed"],
            "estimated_effort": "easy",
        })),
    )
    monkeypatch.setattr(
        "agents.resolution_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({
            "message": "We're on it - please try the steps below.",
            "tone": "reassuring",
        })),
    )


async def test_stream_ticket_endpoint(monkeypatch):
    """POST /tickets/stream emits started/update events and pauses with an interrupt event."""
    _mock_success_pipeline(monkeypatch, triage_data={
        "category": "network", "priority": "high", "business_impact": "high"
    })

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/tickets/stream",
            json={"issue": "network is down", "system": "office router", "impact": "no one can work"},
        ) as response:
            assert response.status_code == 200
            events = await _collect_sse_events(response)

    assert events[0][0] == "started"
    thread_id = events[0][1]["thread_id"]

    update_nodes = [data["node"] for etype, data in events if etype == "update"]
    assert "triage" in update_nodes
    assert "escalate" in update_nodes
    assert "troubleshooting" in update_nodes
    assert "resolution" in update_nodes

    assert events[-1][0] == "interrupt"
    pending = events[-1][1]["pending"]
    assert events[-1][1]["thread_id"] == thread_id
    assert pending["escalated"] is True
    assert pending["resolution"]["message"] == "We're on it - please try the steps below."


async def test_resume_ticket_approve(monkeypatch):
    """Resuming with 'approve' finalizes the ticket and persists it to Mongo."""
    _mock_success_pipeline(monkeypatch, triage_data={
        "category": "hardware", "priority": "low", "business_impact": "low"
    })
    insert_mock = AsyncMock(return_value=SimpleNamespace(inserted_id="fake_id"))
    monkeypatch.setattr("main.tickets_collection.insert_one", insert_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/tickets/stream",
            json={"issue": "mouse not working", "system": "desktop", "impact": "minor annoyance"},
        ) as response:
            events = await _collect_sse_events(response)
        thread_id = events[0][1]["thread_id"]

        resume_response = await client.post(
            f"/tickets/{thread_id}/resume", json={"action": "approve"}
        )

    assert resume_response.status_code == 200
    data = resume_response.json()
    assert data["resolution"]["message"] == "We're on it - please try the steps below."
    insert_mock.assert_awaited_once()


async def test_resume_ticket_edit(monkeypatch):
    """Resuming with 'edit' overrides the resolution message before persisting."""
    _mock_success_pipeline(monkeypatch, triage_data={
        "category": "hardware", "priority": "low", "business_impact": "low"
    })
    insert_mock = AsyncMock(return_value=SimpleNamespace(inserted_id="fake_id"))
    monkeypatch.setattr("main.tickets_collection.insert_one", insert_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/tickets/stream",
            json={"issue": "mouse not working", "system": "desktop", "impact": "minor annoyance"},
        ) as response:
            events = await _collect_sse_events(response)
        thread_id = events[0][1]["thread_id"]

        resume_response = await client.post(
            f"/tickets/{thread_id}/resume",
            json={"action": "edit", "edited_message": "Manually edited resolution."},
        )

    assert resume_response.status_code == 200
    data = resume_response.json()
    assert data["resolution"]["message"] == "Manually edited resolution."
    insert_mock.assert_awaited_once()


async def test_resume_unknown_thread_returns_409():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/tickets/does-not-exist/resume", json={"action": "approve"}
        )
    assert response.status_code == 409