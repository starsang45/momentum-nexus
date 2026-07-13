import uuid

from openai import OpenAIError
from pydantic import BaseModel
from langsmith import traceable

from models.schemas import TicketRequest
from agents.triage_agent import TriageResult
from agents.troubleshooting_agent import TroubleshootingResult
from agents.resolution_agent import ResolutionResult
from agents.graph import graph, TicketGraphState


class TicketResponse(BaseModel):
    """
    Final orchestrated response combining
    all three agent outputs.
    """
    triage: TriageResult
    troubleshooting: TroubleshootingResult
    resolution: ResolutionResult
    escalated: bool = False


@traceable
async def run_ticket_triage(ticket: TicketRequest) -> TicketResponse:
    """
    Run the ticket through the LangGraph pipeline in auto-approve mode
    (no human-in-the-loop pause) and return the combined TicketResponse.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state: TicketGraphState = {
        "ticket": ticket.model_dump(),
        "auto_approve": True,
    }
    final_state = await graph.ainvoke(initial_state, config)

    if final_state.get("failed_node"):
        reason = final_state.get("failure_reason") or "The AI service failed."
        if final_state.get("failure_kind") == "openai":
            raise OpenAIError(reason)
        raise ValueError(reason)

    return TicketResponse(
        triage=TriageResult(**final_state["triage"]),
        troubleshooting=TroubleshootingResult(**final_state["troubleshooting"]),
        resolution=ResolutionResult(**final_state["resolution"]),
        escalated=final_state.get("escalated", False),
    )
