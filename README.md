# Momentum Nexus

# Momentum Nexus

A multi-agent AI IT support ticket triage system powered by FastAPI, OpenAI,
LangGraph, LangSmith, and MongoDB.
An IT support ticket triage backend. A user submits a ticket describing an
issue, the affected system, and its impact; a LangGraph agent workflow
classifies it, plans troubleshooting steps, drafts a customer-facing
resolution message, and can pause for human approval before finalizing.

## How it works

```
TicketRequest (issue, system, impact)
        |
        v
     triage  <---retry (validation/OpenAI error, up to 3 attempts)
        |
        +--(priority=high & business_impact=high)--> escalate --+
        |                                                        |
        +--------------------------------------------------------+
        v
 troubleshooting  <---retry
        |
        v
    resolution  <---retry
        |
        +--(auto_approve=True)------------------------> END
        |
        +--(auto_approve=False)--> human_review (interrupt, waits for approve/edit) --> END
```

The workflow is a LangGraph `StateGraph` (`agents/graph.py`). Each stage
(triage, troubleshooting, resolution) reuses the existing agent functions
(`agents/triage_agent.py`, `troubleshooting_agent.py`, `resolution_agent.py`,
each an OpenAI `gpt-4o` call, traced with LangSmith) but is wrapped in a graph
node with:

- **Conditional branching** — high priority + high business impact tickets
  are routed through an `escalate` node before troubleshooting.
- **Retry loop** — a failed OpenAI call or a malformed/invalid response
  re-enters the same node (self-loop edge) up to 3 attempts before the run
  is marked failed.
- **Human-in-the-loop** — when run via `/tickets/stream`, the graph pauses
  at `human_review` after drafting the resolution message and waits for a
  human to approve or edit it via `/tickets/{thread_id}/resume`. `POST
  /tickets` instead runs in `auto_approve` mode and completes in one call,
  preserving the original synchronous behavior.
- **Streaming** — `/tickets/stream` emits Server-Sent Events as each node
  completes, so a client can show live progress.

State is checkpointed in-memory (`MemorySaver`), which is enough for a
single-process prototype but does not survive a restart or work across
multiple workers — see [Known limitations](#known-limitations).

`agents/orchestrator.py` is now a thin adapter: it invokes the graph in
auto-approve mode and translates a failed run back into the same
`OpenAIError`/`ValueError` exceptions the API layer already expects.
Tickets and their pipeline results are persisted to MongoDB once a run
completes (immediately for `POST /tickets`, or upon resume for the
interactive flow).

## Features

- Multi-agent AI workflow orchestrated as a LangGraph `StateGraph`
- Ticket classification
- Troubleshooting plan generation
- Customer-facing resolution drafting
- Conditional escalation routing (priority + business impact)
- Automatic retry/self-loop on failed or invalid agent responses
- Human-in-the-loop approval/edit of the resolution message
- Real-time progress via Server-Sent Events
- LangSmith tracing
- MongoDB persistence
- FastAPI REST API
- Automated testing with pytest

## Tech Stack

Backend

- FastAPI
- Python

AI

- OpenAI GPT-4o
- LangGraph
- LangSmith

Database

- MongoDB
- Motor

Testing

- pytest
- pytest-asyncio

## Project layout

```
backend/
  agents/
    triage_agent.py           # classify: category / priority / business_impact
    troubleshooting_agent.py  # plan diagnostic steps
    resolution_agent.py       # draft resolution message
    graph.py                  # LangGraph StateGraph: nodes, retry/escalation routing, interrupt
    orchestrator.py           # thin adapter: runs the graph in auto-approve mode
  models/schemas.py           # TicketRequest, ResumeDecision
  database.py                 # MongoDB client + tickets_collection
  main.py                     # FastAPI app and routes
  tests/                      # pytest suite (OpenAI + DB calls mocked)
    test_graph.py              # routing, retry self-loop, escalation, interrupt/resume
  test_db.py                  # manual MongoDB connectivity check
  test_pipeline.py            # manual end-to-end pipeline run
```

## Setup

```bash
cd backend
python -m venv venv
./venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Create `backend/.env`:

```
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=it-support
MONGODB_URL=...
```

## Running the API

```bash
cd backend
uvicorn main:app --reload
```

- `POST /tickets` — submit a ticket, run the full pipeline in auto-approve mode (no pause), persist and return the result
- `GET /tickets` — list the 10 most recent tickets
- `POST /tickets/stream` — submit a ticket and stream progress as Server-Sent Events (`started`, `update`, `progress`, `interrupt`, `failed`); pauses at `human_review` and returns a `thread_id` to resume with
- `POST /tickets/{thread_id}/resume` — resume a ticket paused for review with `{"action": "approve"}` or `{"action": "edit", "edited_message": "..."}`; persists the finalized ticket to MongoDB

## Testing

```bash
cd backend
pytest tests/ -v
```

The test suite mocks OpenAI responses and the MongoDB write, so it runs
offline without incurring API costs. `test_db.py` and `test_pipeline.py` are
standalone manual scripts (not collected by pytest) for checking real
MongoDB/OpenAI connectivity end to end.

## Known limitations

- Graph state is checkpointed with LangGraph's in-memory `MemorySaver`. A
  ticket paused at `human_review` is lost if the process restarts, and
  `/tickets/stream` + `/tickets/{thread_id}/resume` must land on the same
  worker process — fine for local/single-worker use, but a
  Mongo/Postgres-backed checkpointer would be needed to run this with
  multiple `uvicorn` workers or in production.
- The `edit` resume action only overrides the resolution message; it does
  not re-run the resolution agent with human feedback.
