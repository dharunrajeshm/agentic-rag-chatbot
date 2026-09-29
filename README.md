# Agentic AI eBook — RAG Chatbot

A RAG chatbot that answers **only** from the eBook
[*Agentic AI for Executives*](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).
Built with **LangGraph**, **Pinecone**, **OpenAI embeddings + LLM**, and **FastAPI** (plus an optional Streamlit UI).

## Architecture

```
            ┌────────────── Ingestion (python -m app.ingest) ──────────────┐
  PDF ──► pypdf (per page) ──► clean ──► RecursiveCharacterTextSplitter ──► OpenAI embeddings ──► Pinecone
                                          (900 chars, 150 overlap, page metadata kept)

            ┌────────────── Query time (LangGraph) ──────────────┐
  POST /chat ─► [retrieve] ─► top-1 score ≥ MIN_SCORE? ─┬─ yes ─► [generate] ─► answer
                 embed Q +                              │          (strict prompt, temp 0,
                 Pinecone top-k                         └─ no ──► [fallback]  cites [n], NOT_IN_DOCUMENT
                                                                    "not in the eBook"        → fallback)
```

**How answers stay grounded (3 layers)**
1. **Score gate** – if the best retrieved chunk is below `MIN_SCORE`, the LLM is never called; the bot refuses.
2. **Strict prompt** – LLM must use only the numbered context, cite `[n]`, and output `NOT_IN_DOCUMENT` if unsure (temperature 0).
3. **Fallback normalisation** – any `NOT_IN_DOCUMENT` reply is converted into a clean refusal with confidence capped at 0.2.

**Confidence** = heuristic 0–1 from cosine scores: `0.6·top1 + 0.4·mean(top3)`, rescaled (0.25→0, 0.65→1). It measures retrieval support, not a calibrated probability.

## Setup

```bash
git clone <your-repo-url> && cd agentic-rag-chatbot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and PINECONE_API_KEY
```

### 1. Ingest the PDF (one time)
```bash
python -m app.ingest            # downloads the PDF, chunks, embeds, upserts to Pinecone
# or: python -m app.ingest --pdf ./Ebook-Agentic-AI.pdf --reset
```
The Pinecone serverless index (`cosine`, 1536 dims) is created automatically.

### 2. Run the API
```bash
uvicorn app.main:app --reload --port 8000
# docs: http://localhost:8000/docs
```

### 3. (Optional) Chat UI
```bash
streamlit run ui.py
```

## API

`POST /chat`
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "How do LLMs differ from agents?", "top_k": 5}'
```
Response:
```json
{
  "answer": "LLMs are reactive ... agents are goal-driven and act autonomously [1][2].",
  "confidence": 0.82,
  "retrieved_chunks": [
    {"id": "p4-c1", "page": 4, "score": 0.6123, "text": "..."}
  ]
}
```

## Sample queries

| # | Query | Expected behaviour |
|---|-------|--------------------|
| 1 | What is Agentic AI and how is it different from traditional AI? | Grounded answer (ch. 1) |
| 2 | How do LLMs differ from agents? | Answer from the LLM-vs-agents table |
| 3 | What are the categories of agentic systems based on complexity? | Simple reflex, model-based, goal-based, utility-based |
| 4 | What challenges come with orchestrating multi-agent systems? | Communication, conflict mgmt, scalability, fault tolerance |
| 5 | What results did the tire manufacturer case study achieve? | 31% CTR rise, 2.1% conversion growth |
| 6 | Who won the FIFA World Cup in 2022? | Refusal (not in the eBook), low confidence |

Run them all against a live server and generate `sample_outputs.md`:
```bash
python scripts/run_samples.py
```

## Project layout
```
app/config.py      env + constants
app/ingest.py      PDF → chunks → embeddings → Pinecone
app/rag_graph.py   LangGraph pipeline (retrieve / generate / fallback)
app/main.py        FastAPI service
ui.py              Streamlit chat UI
scripts/run_samples.py
```

## Tuning
- `MIN_SCORE` (default 0.30): raise for stricter refusals, lower for more permissive answers.
- `TOP_K`, chunk size/overlap (`app/ingest.py`), `LLM_MODEL`.
- Swap Pinecone for another vector DB by editing `ingest.get_index` and `rag_graph.retrieve`.
