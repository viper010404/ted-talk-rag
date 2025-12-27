"""Ingest the TED dataset: chunk transcripts, embed, and upsert to Pinecone."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import requests
import tiktoken
from pinecone import Pinecone

from config import get_config


def load_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key, val)


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    overlap_ratio = max(0.0, min(overlap_ratio, 0.3))
    chunk_size = max(1, min(chunk_size, 2048))
    
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    overlap = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(enc.decode(chunk_tokens))
    
    return chunks


def batched(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_texts(
    texts: Sequence[str], model: str, api_key: str, base_url: str, timeout: int = 60
) -> List[List[float]]:
    if not texts:
        return []
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": list(texts)}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {resp.status_code} {resp.text}")
    data = resp.json()
    try:
        return [item["embedding"] for item in data["data"]]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unexpected embedding response: {json.dumps(data)}") from exc


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def build_chunks(
    df: pd.DataFrame, chunk_size: int, overlap_ratio: float, limit: int | None
) -> List[Tuple[str, dict]]:
    required_cols = {"talk_id", "title", "transcript"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    records: List[Tuple[str, dict]] = []
    rows = df if limit is None else df.head(limit)
    for _, row in rows.iterrows():
        talk_id = str(row["talk_id"])
        title = str(row["title"])
        url = row.get("url") or ""
        transcript = row.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        chunks = chunk_text(transcript, chunk_size, overlap_ratio)
        for idx, chunk in enumerate(chunks):
            vector_id = f"{talk_id}-{idx}"
            metadata = {
                "talk_id": talk_id,
                "title": title,
                "url": url,
                "chunk_id": idx,
                "text": chunk,
            }
            records.append((vector_id, metadata))
    return records


def upsert_vectors(
    pc: Pinecone,
    host: str,
    vectors: List[Tuple[str, List[float], dict]],
    namespace: str,
    batch_size: int,
) -> None:
    index = pc.Index(host=host)
    for batch in batched(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)


def main() -> None:
    load_dotenv()
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Ingest TED talks into Pinecone")
    parser.add_argument("--dataset", type=Path, default=Path("data/ted_talks_en.csv"))
    parser.add_argument("--limit", type=int, default=5, help="Limit number of talks (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding/upsert batch size")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only, no embeddings")
    args = parser.parse_args()

    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_host = require_env("PINECONE_HOST")
    pinecone_index = os.getenv("PINECONE_INDEX", "ted-talks")
    llm_api_key = os.getenv("MODELS_API_KEY")
    model_base_url = os.getenv("MODEL_BASE_URL")
    if not llm_api_key and not args.dry_run:
        raise RuntimeError("Missing MODELS_API_KEY for embeddings")
    if not model_base_url and not args.dry_run:
        raise RuntimeError("Missing MODEL_BASE_URL for embeddings")

    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset}")

    print("Chunking transcripts...")
    records = build_chunks(df, cfg.chunk_size, cfg.overlap_ratio, args.limit)
    print(f"Prepared {len(records)} chunks")

    if args.dry_run:
        print("Dry run complete; no embeddings or upserts performed.")
        return

    pc = Pinecone(api_key=pinecone_key)
    print(f"Embedding with model {cfg.embedding_model}...")
    vectors: List[Tuple[str, List[float], dict]] = []
    for batch in batched(records, args.batch_size):
        texts = [meta["text"] for _, meta in batch]
        embeds = embed_texts(texts, cfg.embedding_model, llm_api_key, model_base_url)
        if any(len(vec) != cfg.embed_dim for vec in embeds):
            raise RuntimeError("Received embedding with unexpected dimension")
        for (vector_id, meta), vec in zip(batch, embeds):
            vectors.append((vector_id, vec, meta))

    print(f"Upserting {len(vectors)} vectors to index '{pinecone_index}'...")
    upsert_vectors(pc, pinecone_host, vectors, args.namespace, args.batch_size)

    print("Ingestion complete with settings:")
    print(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()
