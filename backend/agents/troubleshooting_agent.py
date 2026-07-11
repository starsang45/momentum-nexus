from openai import AsyncOpenAI
from dotenv import load_dotenv
from models.schemas import TicketRequest
from pydantic import BaseModel
from typing import Literal
import json
from agents.triage_agent import TriageResult
from langsmith import traceable

load_dotenv()

client = AsyncOpenAI()

class TroubleshootingResult(BaseModel):
    """
    Schema for troubleshooting result.
    Capturing diagnostic steps and estimated effort.
    """
    steps: list[str]
    estimated_effort: Literal["easy", "medium", "high"]

@traceable
async def plan_troubleshooting(
    ticket: TicketRequest,
    triage: TriageResult
) -> TroubleshootingResult:
    """
    Analyze the ticket and triage result.
    Return troubleshooting result with realistic diagnostic steps.
    """
    response = await client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an IT support troubleshooting assistant.
                Analyze the ticket and triage result.
                Return ONLY a JSON with:
                -steps: list of 3-5 diagnostic/troubleshooting steps
                -estimated_effort: easy/medium/high
                """
            },
            {
                "role": "user",
                "content": f"""
                Issue:{ticket.issue}
                System:{ticket.system}
                Impact:{ticket.impact}
                Category: {triage.category}
                Priority: {triage.priority}
                Business Impact: {triage.business_impact}
                """
            }
        ],
        response_format = {"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return TroubleshootingResult(**data)