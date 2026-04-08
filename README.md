# rag-llm

          INDEXING (Offline)
Documents → Chunking → Embeddings → Vector DB

          QUERY (Online)
User Query
   ↓
Embedding
   ↓
Retrieval (Top 20)
   ↓
Reranking (Top 5)
   ↓
Compression
   ↓
Prompt
   ↓
LLM
   ↓
Answer


Query
 ↓
Validation (sanitize input)
 ↓
Query rewriting
 ↓
Retrieval (hybrid + filter)
 ↓
MMR (remove duplicates)
 ↓
Reranker
 ↓
Compression
 ↓
Prompt with guardrails
 ↓
LLM
 ↓
Validation (hallucination check)
 ↓
Response / fallback