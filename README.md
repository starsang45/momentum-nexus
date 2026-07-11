# Momentum Nexus

# Momentum Nexus

A multi-agent AI IT support ticket triage system powered by FastAPI, OpenAI, LangSmith, and MongoDB.
An IT support ticket triage backend. A user submits a ticket describing an
issue, the affected system, and its impact; a three-agent pipeline classifies
it, plans troubleshooting steps, and drafts a customer-facing resolution
message.

## How it works

```
TicketRequest (issue, system, impact)
        |
        v
  triage_agent            -> TriageResult (category, priority, business_impact)
        |
        v
  troubleshooting_agent    -> TroubleshootingResult (steps, estimated_effort)
        |
        v
  resolution_agent         -> ResolutionResult (message, tone)
```

Each agent is an OpenAI (`gpt-4o`) call orchestrated in sequence by
`agents/orchestrator.py`, traced with LangSmith. Tickets and their pipeline
results are persisted to MongoDB.

## Features

- Multi-agent AI workflow
- Ticket classification
- Troubleshooting plan generation
- Customer-facing resolution drafting
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
    orchestrator.py           # runs the three agents in sequence
  models/schemas.py           # TicketRequest
  database.py                 # MongoDB client + tickets_collection
  main.py                     # FastAPI app and routes
  tests/                      # pytest suite (OpenAI + DB calls mocked)
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

- `POST /tickets` — submit a ticket, run the pipeline, persist and return the result
- `GET /tickets` — list the 10 most recent tickets

## Testing

```bash
cd backend
pytest tests/ -v
```

The test suite mocks OpenAI responses and the MongoDB write, so it runs
offline without incurring API costs. `test_db.py` and `test_pipeline.py` are
standalone manual scripts (not collected by pytest) for checking real
MongoDB/OpenAI connectivity end to end.
