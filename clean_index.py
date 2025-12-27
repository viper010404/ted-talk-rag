"""Clean Pinecone index by deleting all vectors from a namespace."""

import argparse
import os
from pinecone import Pinecone


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


def main() -> None:
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Clean Pinecone index namespace")
    parser.add_argument("--namespace", type=str, default="default", help="Namespace to delete (default: default)")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    
    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    if not pinecone_host:
        raise RuntimeError("Missing PINECONE_HOST")
    
    if not args.confirm:
        resp = input(f"Delete all vectors from namespace '{args.namespace}'? (yes/no): ")
        if resp.lower() != "yes":
            print("Cancelled.")
            return
    
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(host=pinecone_host)
    
    print(f"Deleting all vectors from namespace '{args.namespace}'...")
    index.delete(delete_all=True, namespace=args.namespace)
    
    print(f"Successfully cleaned namespace '{args.namespace}'")


if __name__ == "__main__":
    main()
