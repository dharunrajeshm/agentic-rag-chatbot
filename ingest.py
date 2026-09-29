"""Ingest the Agentic AI eBook: PDF -> chunks -> embeddings -> Pinecone.

Usage:
    python -m app.ingest                      # downloads PDF from konverge.ai
    python -m app.ingest --pdf path/to.pdf    # use a local copy
    python -m app.ingest --reset              # wipe namespace first
"""
import argparse
import os
import re
import time

import requests
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def get_pdf(path):
    if path:
        return path
    if not os.path.exists(config.PDF_LOCAL_PATH):
        os.makedirs(os.path.dirname(config.PDF_LOCAL_PATH), exist_ok=True)
        print(f"Downloading {config.PDF_URL} ...")
        r = requests.get(config.PDF_URL, timeout=60)
        r.raise_for_status()
        with open(config.PDF_LOCAL_PATH, "wb") as f:
            f.write(r.content)
    return config.PDF_LOCAL_PATH


def clean(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk(pdf_path):
    """Chunk page by page so every chunk keeps its page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    reader = PdfReader(pdf_path)
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if len(text) < 40:  # skip image-only / near-empty pages
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({"id": f"p{page_no}-c{i}", "page": page_no, "text": piece})
    return chunks


def embed(client, texts, batch=100):
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts[i : i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


def get_index(pc):
    if config.PINECONE_INDEX not in pc.list_indexes().names():
        print(f"Creating index {config.PINECONE_INDEX} ...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    return pc.Index(config.PINECONE_INDEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="Local PDF path (default: download from konverge.ai)")
    ap.add_argument("--reset", action="store_true", help="Delete existing vectors in namespace")
    args = ap.parse_args()

    pdf_path = get_pdf(args.pdf)
    chunks = load_and_chunk(pdf_path)
    print(f"Created {len(chunks)} chunks from {pdf_path}")

    oai = OpenAI(api_key=config.OPENAI_API_KEY)
    vectors = embed(oai, [c["text"] for c in chunks])

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = get_index(pc)
    if args.reset:
        try:
            index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
        except Exception:
            pass  # namespace may not exist yet

    records = [
        {
            "id": c["id"],
            "values": v,
            "metadata": {"text": c["text"], "page": c["page"], "source": "Agentic AI for Executives (Konverge AI)"},
        }
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i : i + 100], namespace=config.PINECONE_NAMESPACE)
    print(f"Upserted {len(records)} vectors into {config.PINECONE_INDEX}/{config.PINECONE_NAMESPACE}")


if __name__ == "__main__":
    main()
