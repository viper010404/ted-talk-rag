"""Core RAG pipeline: retrieval, prompt construction, and chat completion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import requests
from pinecone import Pinecone

from config import get_config

SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on the TED "
    "dataset context provided to you (metadata and transcript passages). You must not use any "
    "external knowledge, the open internet, or information that is not explicitly contained in the "
    "retrieved context. If the answer cannot be determined from the provided context, respond: "
    "'I don't know based on the provided TED data.' Always explain your answer using the given "
    "context, quoting or paraphrasing the relevant transcript or metadata when helpful.\n\n"
    "Supported task types (follow these rules when applicable):\n"
    "1) Precise Fact Retrieval: Locate a single, specific entity/fact; provide title and speaker or the exact fact found.\n"
    "2) Multi-Result Topic Listing: Return exactly 3 distinct TED talk titles matching the theme; do not repeat chunks from the same talk; list only titles.\n"
    "3) Key Idea Summary Extraction: Identify a relevant talk and provide a concise summary of its main idea grounded in the retrieved transcript chunk(s).\n"
    "4) Recommendation with Evidence-Based Justification: Recommend one relevant talk and justify the choice using retrieved evidence (cite/quote/paraphrase context).\n\n"
    "General guidelines: Be concise, stick to the retrieved context, avoid speculation, and prefer quoting brief, relevant lines from the transcript when helpful."
)


def load_dotenv(path: str = ".env") -> None:
    """Load environment variables from a .env file if present."""
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
    """Fetch a required environment variable or raise."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


@dataclass
class RAGEnv:
    """Runtime credentials and routing details for the RAG service."""
    models_api_key: str
    model_base_url: str
    pinecone_api_key: str
    pinecone_host: str
    namespace: str = "default"


class RAGService:
    """End-to-end retrieval-augmented generation over the TED dataset."""
    def __init__(self, env: RAGEnv, top_k: int) -> None:
        """Initialize the service with credentials and retrieval depth."""
        self.env = env
        self.top_k = top_k
        self.pc = Pinecone(api_key=env.pinecone_api_key)

    def _embed(self, text: str, model: str) -> List[float]:
        """Call the embedding endpoint to vectorize a query or chunk."""
        url = self.env.model_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.env.models_api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": text}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding request failed: {resp.status_code} {resp.text}")
        data = resp.json()
        return data["data"][0]["embedding"]

    def _retrieve(self, vector: List[float]) -> List[Tuple[str, Dict[str, Any], float]]:
        """Query Pinecone for the nearest chunks and return ids, metadata, and scores."""
        index = self.pc.Index(host=self.env.pinecone_host)
        res = index.query(vector=vector, top_k=self.top_k, include_metadata=True, namespace=self.env.namespace)
        matches = res.get("matches", [])
        return [
            (m.get("id", ""), m.get("metadata", {}), float(m.get("score", 0.0)))
            for m in matches
        ]

    @staticmethod
    def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
        """Format the user-facing prompt that stitches the question with retrieved context."""
        parts = ["Question: " + question, "\nContext:"]
        for idx, ctx in enumerate(contexts, 1):
            chunk = ctx.get("text", "")
            title = ctx.get("title", "")
            talk_id = ctx.get("talk_id", "")
            parts.append(f"[{idx}] talk_id={talk_id} title={title}\n{chunk}\n")
        return "\n".join(parts)

    def _chat(self, model: str, system: str, user: str) -> str:
        """Send the system/user messages to the chat endpoint and return the model reply."""
        url = self.env.model_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.env.models_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 1,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Chat request failed: {resp.status_code} {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def answer(self, question: str) -> Dict[str, Any]:
        """Run embed → retrieve → prompt → chat to answer a user question.

        Args:
            question: Natural-language question to answer using TED content.

        Returns:
            A response payload containing the model answer, context metadata, and the
            augmented prompt used for generation.
        """
        cfg = get_config()
        query_vec = self._embed(question, cfg.embedding_model)
        matches = self._retrieve(query_vec)
        contexts = [m[1] | {"score": m[2]} for m in matches]
        user_prompt = self.build_user_prompt(question, contexts)
        response = self._chat(cfg.chat_model, SYSTEM_PROMPT, user_prompt)
        return {
            "response": response,
            "context": [
                {
                    "talk_id": ctx.get("talk_id", ""),
                    "title": ctx.get("title", ""),
                    "chunk": ctx.get("text", ""),
                    "score": ctx.get("score", 0.0),
                }
                for ctx in contexts
            ],
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt,
            },
        }
