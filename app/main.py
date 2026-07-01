"""
Stage 5: API layer.

Stateless FastAPI service exposing:
  GET  /health -> {"status": "ok"}
  POST /chat   -> takes full conversation history, returns next reply
                  + structured recommendations, per the fixed schema.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal

from app.retrieval import CatalogIndex
from app.agent import run_agent

app = FastAPI(title="SHL Assessment Recommendation Agent")

# Loaded once at startup (not per-request) - embeddings + Groq client are
# expensive/slow to (re)initialize, and doing so per-request would blow
# the 30s-per-call budget.
_index: Optional[CatalogIndex] = None


@app.on_event("startup")
def _load_index():
    global _index
    _index = CatalogIndex()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = run_agent(messages, _index)
    return result
