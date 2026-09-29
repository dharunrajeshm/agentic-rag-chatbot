import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "agentic-ai-ebook")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ebook")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_LOCAL_PATH = "data/Ebook-Agentic-AI.pdf"
