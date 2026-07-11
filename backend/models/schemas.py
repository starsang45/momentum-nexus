from pydantic import BaseModel

class TicketRequest(BaseModel):
    """
    Schema for IT support ticket submission.
    Capture the issue, affected system, and impact from the user.
    """
    issue: str
    system: str
    impact: str