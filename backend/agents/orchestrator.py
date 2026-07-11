from pydantic import BaseModel
from models.schemas import TicketRequest
from agents.triage_agent import triage_ticket, TriageResult
from agents.troubleshooting_agent import plan_troubleshooting, TroubleshootingResult
from agents.resolution_agent import generate_resolution, ResolutionResult
from langsmith import traceable

class TicketResponse(BaseModel):
    """
    Final orchestrated response combining
    all three agent outputs.
    """
    triage: TriageResult
    troubleshooting: TroubleshootingResult
    resolution: ResolutionResult

@traceable
async def run_ticket_triage(ticket: TicketRequest) -> TicketResponse:
    """
    Orchestrate all three agents in sequence.
    Return combined TicketResponse.
    """
    #classify the ticket
    triage = await triage_ticket(ticket)
    #plan troubleshooting steps based on triage
    troubleshooting = await plan_troubleshooting(ticket, triage)
    #generate resolution message based on both
    resolution = await generate_resolution(triage, troubleshooting)

    return TicketResponse(
        triage=triage,
        troubleshooting=troubleshooting,
        resolution=resolution
    )