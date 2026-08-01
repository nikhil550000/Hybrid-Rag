"""Persistent index manifest creation and validation."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1


def manifest_path_for(bm25_index_path: Path) -> Path:
    """Return the manifest path associated with a serialized BM25 index."""
    return bm25_index_path.with_suffix(".manifest.json")


def fingerprint_pdf_files(papers_dir: Path) -> dict[str, str]:
    """Return SHA-256 fingerprints for all PDFs currently in ``papers_dir``."""
    fingerprints: dict[str, str] = {}
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        digest = hashlib.sha256()
        with pdf_path.open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[pdf_path.name] = digest.hexdigest()
    return fingerprints


def digest_chunk_ids(chunk_ids: list[str]) -> str:
    """Create an order-independent digest for the IDs in both indexes."""
    payload = "\n".join(sorted(chunk_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    papers_dir: Path,
    chunk_ids: list[str],
) -> dict[str, Any]:
    """Build the metadata needed to validate a persistent index pair."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_files": fingerprint_pdf_files(papers_dir),
        "chunk_count": len(chunk_ids),
        "chunk_ids_digest": digest_chunk_ids(chunk_ids),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write a manifest next to the persistent indexes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(temporary_path, path)
    logger.info(f"Saved index manifest to {path}")


def validate_manifest(
    *,
    path: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    papers_dir: Path,
    vector_store: Any,
    bm25_store: Any,
) -> None:
    """Fail fast if persistent dense and sparse indexes are not a trusted pair."""
    if not path.exists():
        raise RuntimeError(
            f"Index manifest not found at {path}. Run: python scripts/ingest.py"
        )

    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Index manifest at {path} is unreadable. Run: python scripts/ingest.py"
        ) from exc

    expected_config = {
        "manifest_version": MANIFEST_VERSION,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    mismatches = [
        f"{key}={manifest.get(key)!r} (expected {value!r})"
        for key, value in expected_config.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Persistent index manifest does not match current configuration: "
            + ", ".join(mismatches)
            + ". Run: python scripts/ingest.py"
        )

    current_sources = fingerprint_pdf_files(papers_dir)
    if manifest.get("source_files") != current_sources:
        raise RuntimeError(
            "Persistent index manifest does not match the PDFs in "
            f"{papers_dir}. Run: python scripts/ingest.py"
        )

    vector_ids = vector_store.get_chunk_ids()
    bm25_ids = bm25_store.chunk_ids
    expected_chunk_count = manifest.get("chunk_count")
    if (
        len(vector_ids) != expected_chunk_count
        or len(bm25_ids) != expected_chunk_count
        or set(vector_ids) != set(bm25_ids)
        or digest_chunk_ids(vector_ids) != manifest.get("chunk_ids_digest")
    ):
        raise RuntimeError(
            "Persistent Chroma and BM25 indexes do not contain the same chunks. "
            "Run: python scripts/ingest.py"
        )

    logger.info(
        f"Validated persistent index manifest: {expected_chunk_count} matching chunks"
    )
