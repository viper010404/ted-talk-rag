"""Vercel handler for POST /api/prompt: thin wrapper around rag.RAGService."""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGService, RAGEnv, load_dotenv, require_env
from config import get_config


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
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
        return
