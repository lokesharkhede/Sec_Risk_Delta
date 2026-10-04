"""
Thin wrapper around Hugging Face's free Serverless Inference (via the
OpenAI-compatible router). Docs: https://huggingface.co/docs/huggingface_hub/en/guides/inference

Free tier notes (as of the router's current setup):
  - Base URL: https://router.huggingface.co/v1
  - Auth: your HF access token (Settings -> Access Tokens -> read scope is enough)
  - Rate limit: a few hundred requests/hour on the free tier; upgrading to
    HF PRO ($9/mo) raises this substantially if you hit limits.
  - Any instruct-tuned text-generation model on the Hub works -- swap
    HF_MODEL in .env to try others.
"""
from huggingface_hub import InferenceClient
from google import genai
#from google.genai import types

from config import HF_TOKEN, HF_MODEL, GOOGLE_API_KEY, GEMINI_MODEL

_hf_client = InferenceClient(api_key=HF_TOKEN, model=HF_MODEL) if HF_TOKEN else None
_gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def _gemini_chat(system_prompt: str, user_prompt: str,
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


def chat(system_prompt: str, user_prompt: str, model: str | None = None,
         max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Single-turn chat completion. Returns the assistant's text reply."""
    if _hf_client is None:
        raise RuntimeError(
            "HF_TOKEN not set. Get a free token at "
            "https://huggingface.co/settings/tokens and put it in .env"
        )
    try:
        completion = _hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback to Gemini on any HF failure (e.g. token/context limit exceeded,
        # rate limits, model unavailable, etc.)
        print(f"[chat] Hugging Face call failed ({e}); falling back to Gemini.")
        return _gemini_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )