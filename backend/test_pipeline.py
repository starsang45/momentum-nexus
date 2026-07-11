import asyncio
from agents.triage_agent import triage_ticket
from models.schemas import TicketRequest
from agents.troubleshooting_agent import plan_troubleshooting
from agents.resolution_agent import generate_resolution
from agents.orchestrator import run_ticket_triage

# manual end-to-end pipeline check
async def test():
    ticket = TicketRequest(
        issue="laptop won't boot",
        system="Dell Latitude laptop",
        impact="can't start new job"
    )

    result = await run_ticket_triage(ticket)

    print("=== TICKET RESULT ===")
    print(f"\nTriage:")
    print(f"  Category: {result.triage.category}")
    print(f"  Priority: {result.triage.priority}")
    print(f"  Business Impact: {result.triage.business_impact}")

    print(f"\nTroubleshooting:")
    print(f"  Estimated Effort: {result.troubleshooting.estimated_effort}")
    for i, step in enumerate(result.troubleshooting.steps, 1):
        print(f"  Step {i}: {step}")

    print(f"\nResolution:")
    print(f"  Tone: {result.resolution.tone}")
    print(f"  Message: {result.resolution.message}")

if __name__ == "__main__":
    asyncio.run(test())