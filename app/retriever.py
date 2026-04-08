from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from app.embeddings import get_embedding

CLASS_NAME = "Document"


def create_schema(client):
    if not client.collections.exists(CLASS_NAME):
        client.collections.create(
            name=CLASS_NAME,
            vectorizer_config=None,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ]
        )


def add_document(client, text: str, source: str = "default"):
    vector = get_embedding(text)

    collection = client.collections.get(CLASS_NAME)

    collection.data.insert(
        properties={
            "text": text,
            "source": source
        },
        vector=vector
    )


def hybrid_search(client, query: str, top_k: int = 5):
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