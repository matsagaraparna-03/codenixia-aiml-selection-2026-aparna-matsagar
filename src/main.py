"""
Application / API layer (Milestone 8).

Input -> Processing -> AI/ML (RAG + Agent) -> Output, exposed as a small REST API.
"""

from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
# Load .env explicitly from the project root (one level above this file's folder),
# so it works regardless of which directory the server is launched from.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag
from agent import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the RAG index once at startup rather than on the first request,
    # so the first real user isn't the one who pays the index-build latency.
    key_loaded = bool(os.environ.get("GEMINI_API_KEY"))
    logger.info("GEMINI_API_KEY loaded from .env: %s", key_loaded)
    logger.info("Building RAG index at startup...")
    rag.build_index()
    logger.info("Startup complete.")
    yield


app = FastAPI(
    title="HR/IT Helpdesk Assistant API",
    description="RAG + Agent powered internal helpdesk assistant",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Employee's question")
    employee_id: str = Field(default="UNKNOWN", max_length=50)


class Source(BaseModel):
    source_file: str
    section_title: str
    score: float


class Action(BaseModel):
    tool: str
    input: dict
    result: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    actions_taken: list[Action]


@app.get("/health")
def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Main endpoint: ask the helpdesk assistant a question."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = answer_question(question, employee_id=request.employee_id)
    except RuntimeError as e:
        # Configuration errors (e.g. missing API key) -> 500, not a crash
        logger.error("Configuration error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error handling /ask request")
        raise HTTPException(status_code=500, detail="Internal error processing your question.")

    return result


@app.get("/")
def root():
    return {
        "service": "HR/IT Helpdesk Assistant",
        "endpoints": {"health": "/health", "ask": "POST /ask", "docs": "/docs"},
    }