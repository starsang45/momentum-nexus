import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai import OpenAIError
from langgraph.types import Command

from agents.graph import (
    graph,
    route_after_triage,
    route_after_troubleshooting,
    route_after_resolution,
    MAX_ATTEMPTS,
)


def _fake_openai_response(data: dict):
    """Build a stand-in for an OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))]
    )


def _new_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


TRIAGE_LOW = {"category": "software", "priority": "low", "business_impact": "low"}
TRIAGE_HIGH = {"category": "network", "priority": "high", "business_impact": "high"}
TROUBLESHOOTING_OK = {
    "steps": ["Restart the app", "Clear the cache", "Reinstall if needed"],
    "estimated_effort": "easy",
}
RESOLUTION_OK = {"message": "We're on it.", "tone": "reassuring"}


def _mock_success_pipeline(monkeypatch, triage_data=TRIAGE_LOW):
    monkeypatch.setattr(
        "agents.triage_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(triage_data)),
    )
    monkeypatch.setattr(
        "agents.troubleshooting_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(TROUBLESHOOTING_OK)),
    )
    monkeypatch.setattr(
        "agents.resolution_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(RESOLUTION_OK)),
    )


# --- Router unit tests (no I/O) ---

def test_route_after_triage_retry():
    state = {"triage": None, "triage_attempts": 1}
    assert route_after_triage(state) == "retry"


def test_route_after_triage_fail():
    state = {"triage": None, "triage_attempts": MAX_ATTEMPTS}
    assert route_after_triage(state) == "fail"


def test_route_after_triage_escalate():
    state = {"triage": TRIAGE_HIGH}
    assert route_after_triage(state) == "escalate"


def test_route_after_triage_continue():
    state = {"triage": TRIAGE_LOW}
    assert route_after_triage(state) == "continue"


def test_route_after_troubleshooting_retry():
    assert route_after_troubleshooting({"troubleshooting": None, "troubleshooting_attempts": 1}) == "retry"


def test_route_after_troubleshooting_fail():
    assert route_after_troubleshooting(
        {"troubleshooting": None, "troubleshooting_attempts": MAX_ATTEMPTS}
    ) == "fail"


def test_route_after_troubleshooting_continue():
    assert route_after_troubleshooting({"troubleshooting": TROUBLESHOOTING_OK}) == "continue"


def test_route_after_resolution_retry():
    assert route_after_resolution({"resolution": None, "resolution_attempts": 1}) == "retry"


def test_route_after_resolution_fail():
    assert route_after_resolution({"resolution": None, "resolution_attempts": MAX_ATTEMPTS}) == "fail"


def test_route_after_resolution_finalize_when_auto_approve():
    assert route_after_resolution({"resolution": RESOLUTION_OK, "auto_approve": True}) == "finalize"


def test_route_after_resolution_review_when_not_auto_approve():
    assert route_after_resolution({"resolution": RESOLUTION_OK, "auto_approve": False}) == "review"


# --- Full-graph behavior tests ---

async def test_triage_retry_then_success(monkeypatch):
    """First triage call raises, second succeeds, proving the self-loop retries the node."""
    mock_create = AsyncMock(
        side_effect=[OpenAIError("transient"), _fake_openai_response(TRIAGE_LOW)]
    )
    monkeypatch.setattr("agents.triage_agent.client.chat.completions.create", mock_create)
    monkeypatch.setattr(
        "agents.troubleshooting_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(TROUBLESHOOTING_OK)),
    )
    monkeypatch.setattr(
        "agents.resolution_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response(RESOLUTION_OK)),
    )

    config = _new_config()
    final_state = await graph.ainvoke(
        {"ticket": {"issue": "x", "system": "y", "impact": "z"}, "auto_approve": True},
        config,
    )

    assert final_state["triage_attempts"] == 1
    assert final_state["triage_error_kind"] == "openai"
    assert final_state["triage"]["category"] == "software"
    assert final_state.get("failed_node") is None


async def test_triage_exhausts_retries(monkeypatch):
    """Triage always fails validation; the self-loop must retry MAX_ATTEMPTS times then fail."""
    monkeypatch.setattr(
        "agents.triage_agent.client.chat.completions.create",
        AsyncMock(return_value=_fake_openai_response({"not": "valid"})),
    )

    config = _new_config()
    triage_executions = 0
    final_state = {}
    async for chunk in graph.astream(
        {"ticket": {"issue": "x", "system": "y", "impact": "z"}, "auto_approve": True},
        config,
        stream_mode="updates",
    ):
        if "triage" in chunk:
            triage_executions += 1
        final_state.update(next(iter(chunk.values())))

    assert triage_executions == MAX_ATTEMPTS
    assert final_state["failed_node"] == "triage"
    assert final_state["failure_kind"] == "validation"


async def test_human_in_the_loop_interrupt_and_resume(monkeypatch):
    _mock_success_pipeline(monkeypatch)
    config = _new_config()

    pending_value = None
    async for chunk in graph.astream(
        {"ticket": {"issue": "x", "system": "y", "impact": "z"}, "auto_approve": False},
        config,
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            pending_value = chunk["__interrupt__"][0].value

    assert pending_value is not None
    assert pending_value["resolution"]["message"] == RESOLUTION_OK["message"]

    final_state = await graph.ainvoke(Command(resume={"action": "approve"}), config)
    assert final_state["resolution_approved"] is True
    assert final_state["resolution"]["message"] == RESOLUTION_OK["message"]


async def test_human_in_the_loop_edit_on_resume(monkeypatch):
    _mock_success_pipeline(monkeypatch)
    config = _new_config()

    async for chunk in graph.astream(
        {"ticket": {"issue": "x", "system": "y", "impact": "z"}, "auto_approve": False},
        config,
        stream_mode="updates",
    ):
        pass

    final_state = await graph.ainvoke(
        Command(resume={"action": "edit", "edited_message": "Edited by agent."}),
        config,
    )
    assert final_state["resolution_approved"] is True
    assert final_state["resolution"]["message"] == "Edited by agent."


async def test_escalation_flag_set_for_high_priority_high_impact(monkeypatch):
    _mock_success_pipeline(monkeypatch, triage_data=TRIAGE_HIGH)
    config = _new_config()

    final_state = await graph.ainvoke(
        {"ticket": {"issue": "x", "system": "y", "impact": "z"}, "auto_approve": True},
        config,
    )

    assert final_state["escalated"] is True
