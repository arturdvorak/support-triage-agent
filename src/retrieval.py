"""Similar-case retrieval backed by a local ChromaDB collection.

Cases are loaded once from data/cases.json into an on-disk Chroma collection, then queried by text
embedding (all-MiniLM-L6-v2). Output is a list[SimilarCase], identical in shape
to what the rest of the graph already expects.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing chromadb so vars like HF_HUB_OFFLINE take effect.
# huggingface_hub reads these at import time, so this must run first.
load_dotenv()

import chromadb  # noqa: E402
from chromadb.utils import embedding_functions  # noqa: E402

from src.state import SimilarCase  # noqa: E402

_CASES_PATH = Path(__file__).parent.parent / "data" / "cases.json"
_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
_COLLECTION_NAME = "ear_cases"

# Mirror the phrasing used in data/generate_cases.py so the query embeds close to
# the stored case descriptions. Same "language" -> higher, more meaningful scores.
_DX_TEXT = {
    "normal": "Normal eardrum exam",
    "aom": "Acute otitis media (middle ear infection)",
    "chronic_om": "Chronic otitis media",
}

_CLIENT = chromadb.PersistentClient(path=_DB_PATH)
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection():
    """Return the ear_cases collection, creating it if missing.

    Uses cosine space so distance maps cleanly to similarity (1 - distance).
    """
    return _CLIENT.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def seed_if_empty(collection) -> None:
    """Load data/cases.json into the collection, but only if it has no rows."""
    if collection.count() > 0:
        return

    cases = json.loads(_CASES_PATH.read_text())

    ids = [c["case_id"] for c in cases]
    documents = [c["description"] for c in cases]
    # Chroma metadata values must be scalars, so store symptoms as a joined string.
    metadatas = [
        {
            "case_id": c["case_id"],
            "diagnosis": c["diagnosis"],
            "treatment": c["treatment"],
            "outcome": c["outcome"],
            "age": c["age"],
            "age_band": c["age_band"],
            "symptoms": ", ".join(c["symptoms"]),
        }
        for c in cases
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def _to_similar_case(meta: dict, distance: float) -> SimilarCase:
    # Cosine distance is in [0, 2]; clamp similarity into [0, 1] for a clean score.
    similarity = max(0.0, min(1.0, 1.0 - distance))
    return SimilarCase(
        case_id=meta["case_id"],
        diagnosis=meta["diagnosis"],
        treatment=meta["treatment"],
        outcome=meta["outcome"],
        similarity=round(similarity, 4),
    )


def retrieve_similar(diagnosis: str, age: int, symptoms: list[str], k: int = 3) -> list[SimilarCase]:
    """Return the k most similar past cases that share the given diagnosis."""
    collection = get_collection()
    # Self-seed so direct calls work too, not just the app path. Cheap no-op once seeded.
    seed_if_empty(collection)

    dx_text = _DX_TEXT.get(diagnosis, diagnosis)
    sx = ", ".join(symptoms) if symptoms else "no notable symptoms"
    query_text = f"{dx_text} in a {age}-year-old patient presenting with {sx}."
    res = collection.query(
        query_texts=[query_text],
        n_results=k,
        where={"diagnosis": diagnosis},
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [_to_similar_case(m, d) for m, d in zip(metas, dists)]


def _main() -> None:
    """CLI to inspect retrieval on its own: python -m src.retrieval --diagnosis aom --age 1 --symptoms high_fever"""
    import argparse

    p = argparse.ArgumentParser(description="Query similar past cases directly.")
    p.add_argument("--diagnosis", required=True, help="normal | aom | chronic_om")
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--symptoms", nargs="*", default=[], help="Symptoms, space-separated")
    p.add_argument("-k", type=int, default=3, help="How many cases to return")
    args = p.parse_args()

    hits = retrieve_similar(args.diagnosis, args.age, args.symptoms, k=args.k)
    print(f"\nTop {len(hits)} cases for {args.diagnosis}, age {args.age}, symptoms {args.symptoms or 'none'}:\n")
    for h in hits:
        print(f"  {h.case_id}  sim={h.similarity:.3f}  {h.diagnosis}  | {h.treatment} -> {h.outcome}")


if __name__ == "__main__":
    _main()
