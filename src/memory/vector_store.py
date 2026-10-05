import chromadb
from chromadb.utils import embedding_functions

from config import EMBEDDING_MODEL, CHROMA_DIR

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _collection(ticker: str):
    return _client.get_or_create_collection(
        name=f"risk_factors_{ticker.lower()}",
        embedding_function=_embed_fn,
    )


def store_risk_paragraphs(ticker: str, filing_date: str, accession: str, paragraphs: list[str]):
    if not paragraphs:
        return
    col = _collection(ticker)
    ids = [f"{accession}_{i}" for i in range(len(paragraphs))]
    metadatas = [{"filing_date": filing_date, "accession": accession} for _ in paragraphs]
    # upsert (not add): re-running the same filing must not crash on duplicate IDs
    col.upsert(documents=paragraphs, ids=ids, metadatas=metadatas)


def find_similar_prior_risks(ticker: str, paragraph: str, n_results: int = 3) -> list[dict]:
    col = _collection(ticker)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[paragraph], n_results=min(n_results, col.count()))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "filing_date": meta["filing_date"], "distance": dist})
    return hits
