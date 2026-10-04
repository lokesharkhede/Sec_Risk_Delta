"""
Gemini LLM model for text generation
"""

from google import genai
#from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_MODEL

_gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def chat(system_prompt: str, user_prompt: str,
                  max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Fallback: single-turn chat completion using Gemini."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a key at "
            "https://aistudio.google.com/apikey and put it in .env"
        )
    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={
            "system_instruction":system_prompt,
            "max_output_tokens":max_tokens,
            "temperature":temperature,
        }
    )
    return response.text