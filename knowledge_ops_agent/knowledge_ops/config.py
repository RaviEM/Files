import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
# LLM provider: "ollama" (default, local) or "azure" 
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    #Embeddings + vector store (always local)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    chroma_dir: Path = field (default_factory=lambda: Path( os.getenv("CHROMA_DIR", ROOT / "chroma_db")))
    collection: str = os.getenv("CHROMA_COLLECTION", "enterprise_docs")
    data_dir: Path = field (default_factory=lambda: Path( os.getenv("DATA_DIR", ROOT / "data")))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval / verification thresholds
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.35"))
    grounding_threshold: float = float(os.getenv("GROUNDING_THRESHOLD", "0.8"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # Guardrails
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))

    # Observability
    log_dir: Path= field (default_factory=lambda: Path(os.getenv("LOG_DIR", ROOT / "logs")))

settings = Settings()