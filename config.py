from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RagConfig:
    # RAG hyperparameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1300"))
    overlap_ratio: float = float(os.getenv("OVERLAP_RATIO", "0.15"))
    top_k: int = int(os.getenv("TOP_K", "10"))

    # Models
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1536"))
    chat_model: str = os.getenv("CHAT_MODEL", "RPRTHPB-gpt-5-mini")
    model_base_url: str = os.getenv("MODEL_BASE_URL", "https://api.llmod.ai/v1")

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_env: str = os.getenv("PINECONE_ENV", "")
    pinecone_project: str = os.getenv("PINECONE_PROJECT", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "ted-talks")


def get_config() -> RagConfig:
    """Return current configuration from environment defaults."""
    return RagConfig()
