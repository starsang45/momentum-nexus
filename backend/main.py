import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from agents.orchestrator import run_ticket_triage, TicketResponse
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import TicketRequest, ResumeDecision
from database import tickets_collection
from datetime import datetime, timezone
from openai import OpenAIError
from pydantic import ValidationError
from langgraph.types import Command
from agents.graph import graph
from agents.triage_agent import TriageResult
from agents.troubleshooting_agent import TroubleshootingResult
from agents.resolution_agent import ResolutionResult

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(ticket: TicketRequest):
    try:
        # run agents
        result = await run_ticket_triage(ticket)
    except OpenAIError:
        raise HTTPException(status_code=502, detail="Failed to reach the AI service. Please try again.")
    except (ValueError, ValidationError):
        raise HTTPException(status_code=502, detail="The AI service returned an unexpected response. Please try again.")

    #save to Mongodb
    await tickets_collection.insert_one({
        "issue": ticket.issue,
        "system": ticket.system,
        "impact": ticket.impact,
        "triage": result.triage.model_dump(),
        "troubleshooting": result.troubleshooting.model_dump(),
        "resolution": result.resolution.model_dump(),
        "created_at": datetime.now(timezone.utc)
    })

    return result

@app.get("/tickets")
async def get_tickets():
    tickets = []
    cursor = tickets_collection.find().sort(
        "created_at", -1
    ).limit(10)
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        tickets.append(doc)
    return tickets


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/tickets/stream")
async def stream_ticket(ticket: TicketRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"ticket": ticket.model_dump(), "auto_approve": False}

    async def event_gen():
        yield _sse("started", {"thread_id": thread_id})
        async for mode, chunk in graph.astream(
            initial_state, config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                yield _sse("progress", chunk)
            elif "__interrupt__" in chunk:
                interrupt_obj = chunk["__interrupt__"][0]
                yield _sse("interrupt", {"thread_id": thread_id, "pending": interrupt_obj.value})
            else:
                node_name = next(iter(chunk))
                if node_name.endswith("_failed"):
                    yield _sse("failed", {"thread_id": thread_id, **chunk[node_name]})
                else:
                    yield _sse("update", {"thread_id": thread_id, "node": node_name, "data": chunk[node_name]})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/tickets/{thread_id}/resume", response_model=TicketResponse)
async def resume_ticket(thread_id: str, decision: ResumeDecision):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="No ticket is pending approval for this thread_id.")

    try:
        final_state = await graph.ainvoke(Command(resume=decision.model_dump()), config)
    except OpenAIError:
        raise HTTPException(status_code=502, detail="Failed to reach the AI service. Please try again.")
    except (ValueError, ValidationError):
        raise HTTPException(status_code=502, detail="The AI service returned an unexpected response. Please try again.")

    if final_state.get("failed_node"):
        raise HTTPException(status_code=502, detail=final_state.get("failure_reason", "Resolution failed."))

    result = TicketResponse(
        triage=TriageResult(**final_state["triage"]),
        troubleshooting=TroubleshootingResult(**final_state["troubleshooting"]),
        resolution=ResolutionResult(**final_state["resolution"]),
        escalated=final_state.get("escalated", False),
    )

    await tickets_collection.insert_one({
        "issue": final_state["ticket"]["issue"],
        "system": final_state["ticket"]["system"],
        "impact": final_state["ticket"]["impact"],
        "triage": result.triage.model_dump(),
        "troubleshooting": result.troubleshooting.model_dump(),
        "resolution": result.resolution.model_dump(),
        "escalated": result.escalated,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc)
    })

    return result