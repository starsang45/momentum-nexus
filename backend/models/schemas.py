from typing import Literal
from pydantic import BaseModel, model_validator

class TicketRequest(BaseModel):
    """
    Schema for IT support ticket submission.
    Capture the issue, affected system, and impact from the user.
    """
    issue: str
    system: str
    impact: str


class ResumeDecision(BaseModel):
    """
    Schema for a human's decision when resuming a ticket
    paused for review at the human_review node.
    """
    action: Literal["approve", "edit"]
    edited_message: str | None = None

    @model_validator(mode="after")
    def _validate_edit(self):
        if self.action == "edit" and not self.edited_message:
            raise ValueError("edited_message is required when action is 'edit'")
        return self