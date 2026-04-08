from sentence_transformers import CrossEncoder

# Load once (important for performance)
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, documents):
    pairs = [(query, doc["text"]) for doc in documents]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked]