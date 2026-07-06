"""Settings loader — single source of truth for all configuration.

Satisfies: FR-33, FR-34, FR-35, NFR-07
"""
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Settings:
    """All configuration loaded from settings.yaml + environment."""
    # LLM
    llm_provider: str
    llm_model: str
    llm_temperature: float
    # Embeddings
    embedding_model: str
    # Retrieval
    retrieval_top_k: int
    reranked_top_n: int
    rrf_k: int
    # Reranker
    reranker_model: str
    reranker_provider: str
    # Chunking
    chunk_size: int
    chunk_overlap: int
    # Prompts
    prompt_version: str
    # Storage
    vector_store_path: str
    bm25_index_path: str
    collection_name: str
    # Eval
    eval_dataset_path: str
    faithfulness_threshold: float
    # Observability
    observability_backend: str
    log_level: str


def load_settings(config_path: Path = Path("config/settings.yaml")) -> Settings:
    """
    Load settings from YAML. API keys come from environment (load_dotenv called here).
    Raises FileNotFoundError if config_path doesn't exist — fails fast, no silent defaults.
    """
    load_dotenv()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Expected at project root: config/settings.yaml"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    settings = Settings(
        llm_provider=raw["llm"]["provider"],
        llm_model=raw["llm"]["model"],
        llm_temperature=raw["llm"]["temperature"],
        embedding_model=raw["embeddings"]["model"],
        retrieval_top_k=raw["retrieval"]["top_k"],
        reranked_top_n=raw["retrieval"]["top_k_reranked"],
        rrf_k=raw["retrieval"]["rrf_k"],
        reranker_model=raw["reranker"]["model"],
        reranker_provider=raw["reranker"]["provider"],
        chunk_size=raw["chunking"]["chunk_size"],
        chunk_overlap=raw["chunking"]["chunk_overlap"],
        prompt_version=raw["prompts"]["version"],
        vector_store_path=raw["store"]["vector_store_path"],
        bm25_index_path=raw["store"]["bm25_index_path"],
        collection_name=raw["store"]["collection_name"],
        eval_dataset_path=raw["eval"]["dataset_path"],
        faithfulness_threshold=raw["eval"]["faithfulness_threshold"],
        observability_backend=raw["observability"]["backend"],
        log_level=raw["observability"]["log_level"],
    )

    logger.info(f"Settings loaded from {config_path}")
    return settings
