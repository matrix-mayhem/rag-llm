from fastapi import FastAPI
from contextlib import asynccontextmanager
import weaviate
import os

from app.retriever import create_schema, hybrid_search, add_document
from app.llm import generate_answer
from app.reranker import rerank

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
    client = app.state.weaviate

    docs = hybrid_search(client, q, top_k=10)

    docs = rerank(q, docs)[:5]

    docs = deduplicate(docs)

    answer = generate_answer(q, docs)

    return {
        "answer": answer,
        "sources": docs
    }