"""
Central configuration. Every other module imports from here so there's
exactly one place to change endpoints, models, or paths.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---- SEC EDGAR (free, no key, but requires a User-Agent header) ----
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "SEC-Risk-Delta-Agent demo@example.com")
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_COMPANYCONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
SEC_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_RATE_LIMIT_SLEEP_SEC = 0.12   # keeps us under the 10 req/sec fair-access ceiling

# ---- Hugging Face free Inference (serverless router, OpenAI-compatible) ------> In case HF is used
#HF_TOKEN = os.getenv("HF_TOKEN", "")
#HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
# Any instruct model on HF's free router works. Swap freely -- e.g.
# "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3".
#HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-14B")

# ----Gemini free Inference ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # or gemini-1.5-pro, etc.

# ---- Local storage ----
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma")

# ---- Embedding Model ----
EMBEDDING_MODEL ='all-MiniLM-L6-v2'

# ---- Risk-factor section identifiers we diff release-over-release ----
TARGET_SECTIONS = {
    "risk_factors": ["item 1a", "risk factors"],
    "legal_proceedings": ["item 3", "legal proceedings"],
    "mdna": ["item 7", "management's discussion and analysis"],
}
