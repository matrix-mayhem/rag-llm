import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from app.embeddings import get_embedding
import os

CLASS_NAME = "Document"


def get_client():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=8080,
        grpc_port=50051
    )


def create_schema():
    client = get_client()

    if not client.collections.exists(CLASS_NAME):
        client.collections.create(
            name=CLASS_NAME,
            vectorizer_config=None,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ]
        )


def add_document(text: str, source: str = "default"):
    client = get_client()
    vector = get_embedding(text)

    docs = client.collections.get(CLASS_NAME)

    docs.insert(
        properties={"text": text, "source": source},
        vector=vector
    )


def hybrid_search(query: str, top_k: int = 5):
    client = get_client()
    vector = get_embedding(query)

    docs = client.collections.get(CLASS_NAME)

    result = docs.query.hybrid(
        query=query,
        vector=vector,
        alpha=0.5,
        limit=top_k,
        filters=Filter.by_property("source").equal("default"),
        return_properties=["text", "source"]
    )

    return [obj.properties for obj in result.objects]