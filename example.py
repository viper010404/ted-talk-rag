"""
Quick Pinecone connectivity check:
- Reads API key, host, and index name from env.
- Upserts a test vector with the correct dimension.
- Fetches it back to verify.
"""

import os
import sys
from pinecone import Pinecone


def load_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader to avoid missing env var errors."""
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
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val


def main() -> None:
    load_dotenv()
    api_key = require_env("PINECONE_API_KEY")
    host = require_env("PINECONE_HOST")
    index_name = require_env("PINECONE_INDEX")
    dim = int(os.getenv("EMBED_DIM", "1536"))

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=host)

    # quick health check
    stats = index.describe_index_stats()
    print("Index stats:", stats)

    # upsert a test vector with the right dimension
    test_id = "ping-1"
    # must include non-zero values; Pinecone rejects all-zero vectors
    vec = [0.001] * dim
    index.upsert(vectors=[(test_id, vec, {"note": "ping", "index": index_name})])
    print("Upserted", test_id)

    # fetch it back
    fetched = index.fetch(ids=[test_id])
    print("Fetched:", fetched)

    # clean up the test vector to avoid clutter
    index.delete(ids=[test_id])
    print("Deleted test vector", test_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
