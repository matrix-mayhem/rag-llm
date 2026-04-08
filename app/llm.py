from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(query, context_chunks):
    context = "\n".join([c["text"] for c in context_chunks])

    prompt = f"""
    Answer strictly based on content

    Context:
    {context}

    Query:
    {query}
    """
    generate_config = types.GenerateContentConfig(
    system_instruction="You are a sarcastic but brilliant technical architect.",
    temperature=0.7,
    top_p=0.95,
    max_output_tokens=256,
    stop_sequences=["STOP"]
    )

    response = client.models.generate_content_stream(
        model = "gemini-2.5-flash",
        contents = prompt,
        config=generate_config
    )

    return response.text