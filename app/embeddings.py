from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_embedding(text: str):
    result = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = text
    )
    return result.embeddings[0].values