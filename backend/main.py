from fastapi import FastAPI, HTTPException
from agents.orchestrator import run_ticket_triage, TicketResponse
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import TicketRequest
from database import tickets_collection
from datetime import datetime, timezone
from openai import OpenAIError
from pydantic import ValidationError

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