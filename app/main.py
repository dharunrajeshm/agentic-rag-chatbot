from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_graph import ask

app = FastAPI(title="Agentic AI eBook RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is Agentic AI?"])
    top_k: int = Field(5, ge=1, le=10)


class Chunk(BaseModel):
    id: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float = Field(..., description="0-1 heuristic derived from retrieval similarity scores")
    retrieved_chunks: List[Chunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return ask(req.question, req.top_k)
    except Exception as e:  # surface config/API errors clearly
        raise HTTPException(status_code=500, detail=str(e))
