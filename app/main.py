from fastapi import FastAPI
from contextlib import asynccontextmanager
import weaviate
import os
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from google import genai

from app.retriever import create_schema, hybrid_search, add_document
from app.llm import generate_answer
from app.reranker import rerank
from app.cache import get_cache,set_cache
from app.token_tracker import count_tokens

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def deduplicate(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            unique_docs.append(doc)

    return unique_docs

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up...")

    # ✅ Create ONE shared Weaviate client
    client = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=8080,
        grpc_port=50051
    )

    # Store in app state
    app.state.weaviate = client

    create_schema(client)

    yield

    # ✅ Proper cleanup (fixes memory leak warning)
    print("🛑 Shutting down...")
    client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "RAG system running"}


@app.post("/ingest")
def ingest(text: str, source: str = "default"):
    add_document(app.state.weaviate, text, source)
    return {"status": "stored"}

@app.get("/query")
def query(q: str):
    cache_key = f"query:{q}"

    # ✅ Check cache first
    cached = get_cache(cache_key)
    if cached:
        return {
            "answer": cached,
            "cached": True
        }

    client = app.state.weaviate

    docs = hybrid_search(client, q, top_k=10)
    docs = rerank(q, docs)[:5]
    docs = deduplicate(docs)

    answer = generate_answer(q, docs)

    # ✅ Store in cache
    set_cache(cache_key, answer)

    return {
        "answer": answer,
        "cached": False,
        "sources": docs
    }

@app.get("/stream")
def stream(q: str):
    client = app.state.weaviate

    docs = hybrid_search(client, q, top_k=10)
    docs = rerank(q, docs)[:5]

    context = "\n".join([c["text"] for c in docs])

    prompt = f"""
    Answer strictly based on the context.

    Context:
    {context}

    Question:
    {q}
    """

    # 🔥 New explicit streaming API
    stream = gemini_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return StreamingResponse(
        (chunk.text for chunk in stream if chunk.text),
        media_type="text/plain"
    )
