"""POST /api/prompt endpoint handler.

Accepts JSON payload with 'question' only. Server enforces configured
retrieval depth (top_k) and does not accept client overrides.
Returns RAG-augmented response with retrieved context.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGService, RAGEnv, load_dotenv, require_env
from config import get_config


class handler(BaseHTTPRequestHandler):
    """HTTP handler for RAG prompt requests."""
    def do_POST(self):
        """Validate input, run RAG, and return the response payload."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid json"}).encode())
            return

        question = (data.get("question") or "").strip()
        if not question:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "missing question"}).encode())
            return

        try:
            load_dotenv()
            cfg = get_config()

            env = RAGEnv(
                models_api_key=require_env("MODELS_API_KEY"),
                model_base_url=require_env("MODEL_BASE_URL"),
                pinecone_api_key=require_env("PINECONE_API_KEY"),
                pinecone_host=require_env("PINECONE_HOST"),
                namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            )
            # Do not allow client overrides of top_k; enforce config value.
            service = RAGService(env=env, top_k=cfg.top_k)
            result = service.answer(question)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
