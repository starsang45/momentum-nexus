from openai import AsyncOpenAI
from dotenv import load_dotenv
from models.schemas import TicketRequest
from pydantic import BaseModel
from typing import Literal
import json
from langsmith import traceable

load_dotenv()

client = AsyncOpenAI()

class TriageResult(BaseModel):
    """
    Schema for ticket triage output.
    Capturing category, priority, and business impact.
    """
    category: Literal["hardware", "software", "network", "account", "other"]
    priority: Literal["low", "medium", "high"]
    business_impact: Literal["low", "medium", "high"]

@traceable
async def triage_ticket(ticket: TicketRequest) -> TriageResult:
    """
    Analyze an IT support ticket.
    Return category, priority, and business impact.
    """
    response = await client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an IT support ticket triage assistant.
                Analyze the user's issue, affected system, and impact.
                Return ONLY a JSON with:
                -category: hardware/software/network/account/other
                -priority: low/medium/high
                -business_impact: low/medium/high
                """
            },
            {
                "role": "user",
                "content": f"""
                Issue:{ticket.issue}
                System:{ticket.system}
                Impact:{ticket.impact}
                """
            }
        ],
        response_format = {"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return TriageResult(**data)