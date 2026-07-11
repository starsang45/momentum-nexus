from openai import AsyncOpenAI
from dotenv import load_dotenv
from models.schemas import TicketRequest
from pydantic import BaseModel
from typing import Literal
import json
from agents.triage_agent import TriageResult
from agents.troubleshooting_agent import TroubleshootingResult
from langsmith import traceable

load_dotenv()

client = AsyncOpenAI()

class ResolutionResult(BaseModel):
    """
    Schema for resolution result output.
    Capturing customer-facing message and tone.
    """
    message: str
    tone: Literal["reassuring", "informative", "urgent"]

@traceable
async def generate_resolution(
        triage: TriageResult,
        troubleshooting: TroubleshootingResult
        ) -> ResolutionResult:
    """
    Generate a customer-facing resolution message from triage and troubleshooting results.
    Return the message and its tone.
    """
    response = await client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an IT support resolution message generator.
                Generate a customer-facing message based on priority and business impact.
                Return ONLY a JSON with:
                -message: str
                -tone: reassuring/informative/urgent
                """
            },
            {
                "role": "user",
                "content": f"""
                Category:{triage.category}
                Priority:{triage.priority}
                Business_impact:{triage.business_impact}
                Steps:{troubleshooting.steps}
                Estimated_effort:{troubleshooting.estimated_effort}
                """
            }
        ],
        response_format = {"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return ResolutionResult(**data)