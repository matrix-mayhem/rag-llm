from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from app.token_tracker import count_tokens

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

    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
        config=generate_config
    )

    input_tokens = count_tokens(prompt)

    output = response.text
    output_tokens = count_tokens(output)

    print(f"Tokens → Input: {input_tokens}, Output: {output_tokens}")

    return output