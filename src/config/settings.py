"""Settings loader — single source of truth for all configuration.

Satisfies: FR-33, FR-34, FR-35, NFR-07
"""
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_LLM_PROVIDERS = {"anthropic", "google"}
SUPPORTED_EVAL_JUDGE_PROVIDERS = {"anthropic", "google"}
SUPPORTED_RERANKER_PROVIDERS = {"sentence_transformers"}


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
    eval_judge_provider: str
    eval_judge_model: str
    # Observability
    observability_backend: str
    log_level: str


def _require_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}. Got: {value!r}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got: {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer. Got: {value!r}")


def _require_project_relative_path(name: str, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty relative path")

    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the project directory. Got: {value!r}")

    parent = path.parent
    if str(parent) not in {"", "."} and not parent.exists():
        raise ValueError(f"{name} parent directory does not exist: {parent}")

    return path


def _validate_prompt_version(prompt_version: str) -> None:
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompts.version must be a non-empty string")
    if Path(prompt_version).is_absolute() or ".." in Path(prompt_version).parts:
        raise ValueError(f"prompts.version must stay inside prompts/. Got: {prompt_version!r}")

    prompts_dir = Path("prompts") / prompt_version
    missing = [
        str(path)
        for path in (prompts_dir / "system.txt", prompts_dir / "query.txt")
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"Prompt files missing for version {prompt_version!r}: {', '.join(missing)}")


def validate_settings(settings: Settings) -> None:
    """Fail fast on config values that would otherwise break later at runtime."""
    _require_choice("llm.provider", settings.llm_provider, SUPPORTED_LLM_PROVIDERS)
    _require_choice(
        "eval.judge_provider",
        settings.eval_judge_provider,
        SUPPORTED_EVAL_JUDGE_PROVIDERS,
    )
    _require_choice(
        "reranker.provider",
        settings.reranker_provider,
        SUPPORTED_RERANKER_PROVIDERS,
    )

    _require_positive_int("retrieval.top_k", settings.retrieval_top_k)
    _require_positive_int("retrieval.top_k_reranked", settings.reranked_top_n)
    _require_positive_int("retrieval.rrf_k", settings.rrf_k)
    _require_positive_int("chunking.chunk_size", settings.chunk_size)
    _require_non_negative_int("chunking.chunk_overlap", settings.chunk_overlap)
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            "chunking.chunk_overlap must be smaller than chunking.chunk_size. "
            f"Got overlap={settings.chunk_overlap}, size={settings.chunk_size}"
        )

    _validate_prompt_version(settings.prompt_version)
    _require_project_relative_path("store.vector_store_path", settings.vector_store_path)
    _require_project_relative_path("store.bm25_index_path", settings.bm25_index_path)
    eval_dataset_path = _require_project_relative_path("eval.dataset_path", settings.eval_dataset_path)
    if not eval_dataset_path.is_file():
        raise ValueError(f"eval.dataset_path does not exist: {eval_dataset_path}")


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
        eval_judge_provider=raw["eval"].get("judge_provider", raw["llm"]["provider"]),
        eval_judge_model=raw["eval"].get("judge_model", raw["llm"]["model"]),
        observability_backend=raw["observability"]["backend"],
        log_level=raw["observability"]["log_level"],
    )

    validate_settings(settings)

    logger.info(f"Settings loaded from {config_path}")
    return settings
