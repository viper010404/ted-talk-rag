"""GET /api/stats endpoint handler.

Returns current RAG hyperparameters (chunk_size, overlap_ratio, top_k).
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config


class handler(BaseHTTPRequestHandler):
    """HTTP handler for reporting current RAG hyperparameters."""
    def do_GET(self):
        """Return chunking and retrieval settings as JSON."""
        cfg = get_config()
        response = {
            "chunk_size": cfg.chunk_size,
            "overlap_ratio": cfg.overlap_ratio,
            "top_k": cfg.top_k,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
