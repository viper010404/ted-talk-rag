"""Ingest TED talks: chunk, embed, and upsert to Pinecone with checkpointing."""

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
from tqdm import tqdm

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


def load_progress(path: Path) -> set[str]:
    """Read processed talk_ids from a checkpoint file."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_progress(path: Path, talk_id: str) -> None:
    """Append a processed talk_id to the checkpoint file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{talk_id}\n")


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """Tokenize text and return overlapping chunks sized for embeddings."""
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
    """Yield a sequence in fixed-size batches."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_texts(
    texts: Sequence[str], model: str, api_key: str, base_url: str, timeout: int = 60
) -> List[List[float]]:
    """Request embeddings for a list of texts from the configured model API."""
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
    """Load the TED talks CSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)



def upsert_vectors(
    pc: Pinecone,
    host: str,
    vectors: List[Tuple[str, List[float], dict]],
    namespace: str,
    batch_size: int,
) -> None:
    """Upsert vector batches into Pinecone."""
    index = pc.Index(host=host)
    for batch in batched(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)


def main() -> None:
    """CLI entry point to chunk, embed, and ingest TED talks with checkpointing.

    Parses command-line options, batches embeddings and upserts to Pinecone, and
    persists a progress file so reruns skip already ingested talks.
    """
    load_dotenv()
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Ingest TED talks into Pinecone")
    parser.add_argument("--dataset", type=Path, default=Path("data/ted_talks_en.csv"))
    parser.add_argument("--limit", type=int, default=5, help="Limit number of talks (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding/upsert batch size")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Upsert and checkpoint every N talks")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only, no embeddings")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("ingest_progress.txt"),
        help="Checkpoint file to skip already processed talks",
    )
    args = parser.parse_args()

    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_host = require_env("PINECONE_HOST")
    llm_api_key = os.getenv("MODELS_API_KEY")
    model_base_url = os.getenv("MODEL_BASE_URL")
    if not llm_api_key and not args.dry_run:
        raise RuntimeError("Missing MODELS_API_KEY for embeddings")
    if not model_base_url and not args.dry_run:
        raise RuntimeError("Missing MODEL_BASE_URL for embeddings")

    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset}")

    # Progress tracking
    processed = load_progress(args.progress_file)
    if processed:
        print(f"Found {len(processed)} processed talks in {args.progress_file}; they will be skipped")

    rows = df if args.limit is None else df.head(args.limit)
    total_chunks = 0
    ingested_talks = 0
    pending_records: List[Tuple[str, dict]] = []
    pending_talk_ids: List[str] = []

    pc = Pinecone(api_key=pinecone_key) if not args.dry_run else None

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Talks"):
        talk_id = str(row["talk_id"])
        if talk_id in processed:
            continue

        title = str(row["title"])
        speaker = str(row.get("speaker_1") or "")
        topics = str(row.get("topics") or "")
        description = str(row.get("description") or "")
        recorded_date = str(row.get("recorded_date") or "")
        native_lang = str(row.get("native_lang") or "")
        url = str(row.get("url") or "")
        transcript = row.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            continue

        chunks = chunk_text(transcript, cfg.chunk_size, cfg.overlap_ratio)
        if not chunks:
            continue

        # Build records for this talk
        for idx, chunk in enumerate(chunks):
            vector_id = f"{talk_id}-{idx}"
            metadata = {
                "talk_id": talk_id,
                "title": title,
                "speaker": speaker,
                "topics": topics,
                "description": description,
                "recorded_date": recorded_date,
                "native_lang": native_lang,
                "url": url,
                "chunk_id": idx,
                "text": chunk,
            }
            pending_records.append((vector_id, metadata))

        pending_talk_ids.append(talk_id)
        total_chunks += len(chunks)

        # Checkpoint: embed and upsert every N talks
        if not args.dry_run and len(pending_talk_ids) >= args.checkpoint_every:
            vectors: List[Tuple[str, List[float], dict]] = []
            for batch in batched(pending_records, args.batch_size):
                texts = [meta["text"] for _, meta in batch]
                embeds = embed_texts(texts, cfg.embedding_model, llm_api_key, model_base_url)
                if any(len(vec) != cfg.embed_dim for vec in embeds):
                    raise RuntimeError("Received embedding with unexpected dimension")
                for (vector_id, meta), vec in zip(batch, embeds):
                    vectors.append((vector_id, vec, meta))

            upsert_vectors(pc, pinecone_host, vectors, args.namespace, args.batch_size)
            for tid in pending_talk_ids:
                append_progress(args.progress_file, tid)
            ingested_talks += len(pending_talk_ids)
            pending_records.clear()
            pending_talk_ids.clear()

    # Final flush for remaining talks
    if not args.dry_run and pending_records:
        vectors: List[Tuple[str, List[float], dict]] = []
        for batch in batched(pending_records, args.batch_size):
            texts = [meta["text"] for _, meta in batch]
            embeds = embed_texts(texts, cfg.embedding_model, llm_api_key, model_base_url)
            if any(len(vec) != cfg.embed_dim for vec in embeds):
                raise RuntimeError("Received embedding with unexpected dimension")
            for (vector_id, meta), vec in zip(batch, embeds):
                vectors.append((vector_id, vec, meta))

        upsert_vectors(pc, pinecone_host, vectors, args.namespace, args.batch_size)
        for tid in pending_talk_ids:
            append_progress(args.progress_file, tid)
        ingested_talks += len(pending_talk_ids)

    if args.dry_run:
        print(f"Dry run complete; prepared {total_chunks} chunks across {len(rows)} talks.")
        return

    print(f"Ingestion complete. Talks ingested: {ingested_talks}, chunks upserted: {total_chunks}")
    print(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()
