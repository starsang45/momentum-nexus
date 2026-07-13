from typing import Literal, TypedDict

from openai import OpenAIError
from pydantic import ValidationError

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.config import get_stream_writer

from models.schemas import TicketRequest
from agents.triage_agent import triage_ticket, TriageResult
from agents.troubleshooting_agent import plan_troubleshooting, TroubleshootingResult
from agents.resolution_agent import generate_resolution, ResolutionResult

MAX_ATTEMPTS = 3


class TicketGraphState(TypedDict, total=False):
    ticket: dict
    auto_approve: bool

    triage: dict | None
    triage_attempts: int
    triage_error: str | None
    triage_error_kind: Literal["openai", "validation"] | None

    escalated: bool

    troubleshooting: dict | None
    troubleshooting_attempts: int
    troubleshooting_error: str | None
    troubleshooting_error_kind: Literal["openai", "validation"] | None

    resolution: dict | None
    resolution_attempts: int
    resolution_error: str | None
    resolution_error_kind: Literal["openai", "validation"] | None
    resolution_approved: bool

    failed_node: str | None
    failure_kind: Literal["openai", "validation"] | None
    failure_reason: str | None


def _notify(node: str, status: str) -> None:
    get_stream_writer()({"node": node, "status": status})


async def triage_node(state: TicketGraphState) -> dict:
    _notify("triage", "in_progress")
    ticket = TicketRequest(**state["ticket"])
    try:
        result = await triage_ticket(ticket)
    except OpenAIError as e:
        return {
            "triage_attempts": state.get("triage_attempts", 0) + 1,
            "triage_error": str(e),
            "triage_error_kind": "openai",
        }
    except (ValueError, ValidationError) as e:
        return {
            "triage_attempts": state.get("triage_attempts", 0) + 1,
            "triage_error": str(e),
            "triage_error_kind": "validation",
        }
    return {"triage": result.model_dump(), "triage_error": None}


def route_after_triage(state: TicketGraphState) -> str:
    if state.get("triage") is None:
        if state.get("triage_attempts", 0) >= MAX_ATTEMPTS:
            return "fail"
        return "retry"
    triage = state["triage"]
    if triage["priority"] == "high" and triage["business_impact"] == "high":
        return "escalate"
    return "continue"


async def escalate_node(state: TicketGraphState) -> dict:
    _notify("escalate", "in_progress")
    return {"escalated": True}


async def troubleshooting_node(state: TicketGraphState) -> dict:
    _notify("troubleshooting", "in_progress")
    ticket = TicketRequest(**state["ticket"])
    triage = TriageResult(**state["triage"])
    try:
        result = await plan_troubleshooting(ticket, triage)
    except OpenAIError as e:
        return {
            "troubleshooting_attempts": state.get("troubleshooting_attempts", 0) + 1,
            "troubleshooting_error": str(e),
            "troubleshooting_error_kind": "openai",
        }
    except (ValueError, ValidationError) as e:
        return {
            "troubleshooting_attempts": state.get("troubleshooting_attempts", 0) + 1,
            "troubleshooting_error": str(e),
            "troubleshooting_error_kind": "validation",
        }
    return {"troubleshooting": result.model_dump(), "troubleshooting_error": None}


def route_after_troubleshooting(state: TicketGraphState) -> str:
    if state.get("troubleshooting") is None:
        if state.get("troubleshooting_attempts", 0) >= MAX_ATTEMPTS:
            return "fail"
        return "retry"
    return "continue"


async def resolution_node(state: TicketGraphState) -> dict:
    _notify("resolution", "in_progress")
    triage = TriageResult(**state["triage"])
    troubleshooting = TroubleshootingResult(**state["troubleshooting"])
    try:
        result = await generate_resolution(triage, troubleshooting)
    except OpenAIError as e:
        return {
            "resolution_attempts": state.get("resolution_attempts", 0) + 1,
            "resolution_error": str(e),
            "resolution_error_kind": "openai",
        }
    except (ValueError, ValidationError) as e:
        return {
            "resolution_attempts": state.get("resolution_attempts", 0) + 1,
            "resolution_error": str(e),
            "resolution_error_kind": "validation",
        }
    return {"resolution": result.model_dump(), "resolution_error": None}


def route_after_resolution(state: TicketGraphState) -> str:
    if state.get("resolution") is None:
        if state.get("resolution_attempts", 0) >= MAX_ATTEMPTS:
            return "fail"
        return "retry"
    return "finalize" if state.get("auto_approve") else "review"


async def human_review_node(state: TicketGraphState) -> dict:
    decision = interrupt(
        {
            "ticket": state["ticket"],
            "triage": state["triage"],
            "troubleshooting": state["troubleshooting"],
            "resolution": state["resolution"],
            "escalated": state.get("escalated", False),
        }
    )
    if decision.get("action") == "edit":
        edited = {**state["resolution"], "message": decision["edited_message"]}
        return {"resolution": edited, "resolution_approved": True}
    return {"resolution_approved": True}


def make_failed_node(stage: str):
    error_key = f"{stage}_error"
    kind_key = f"{stage}_error_kind"

    async def _node(state: TicketGraphState) -> dict:
        return {
            "failed_node": stage,
            "failure_kind": state.get(kind_key, "validation"),
            "failure_reason": state.get(error_key, f"{stage} failed after {MAX_ATTEMPTS} attempts"),
        }

    return _node


def build_graph() -> StateGraph:
    builder = StateGraph(TicketGraphState)

    builder.add_node("triage", triage_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("troubleshooting", troubleshooting_node)
    builder.add_node("resolution", resolution_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("triage_failed", make_failed_node("triage"))
    builder.add_node("troubleshooting_failed", make_failed_node("troubleshooting"))
    builder.add_node("resolution_failed", make_failed_node("resolution"))

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "retry": "triage",
            "escalate": "escalate",
            "continue": "troubleshooting",
            "fail": "triage_failed",
        },
    )
    builder.add_edge("escalate", "troubleshooting")
    builder.add_conditional_edges(
        "troubleshooting",
        route_after_troubleshooting,
        {
            "retry": "troubleshooting",
            "continue": "resolution",
            "fail": "troubleshooting_failed",
        },
    )
    builder.add_conditional_edges(
        "resolution",
        route_after_resolution,
        {
            "retry": "resolution",
            "review": "human_review",
            "finalize": END,
            "fail": "resolution_failed",
        },
    )
    builder.add_edge("human_review", END)
    builder.add_edge("triage_failed", END)
    builder.add_edge("troubleshooting_failed", END)
    builder.add_edge("resolution_failed", END)

    return builder


checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)
