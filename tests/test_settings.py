from pathlib import Path

import pytest
import yaml

from config.settings import load_settings


def _write_config(tmp_path, mutate):
    raw = yaml.safe_load(Path("config/settings.yaml").read_text())
    mutate(raw)
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def test_load_settings_accepts_current_project_config():
    settings = load_settings()

    assert settings.llm_provider in {"anthropic", "google"}
    assert settings.retrieval_top_k > 0


def test_load_settings_rejects_unsupported_llm_provider(tmp_path):
    config_path = _write_config(
        tmp_path,
        lambda raw: raw["llm"].update({"provider": "openai"}),
    )

    with pytest.raises(ValueError, match="llm.provider"):
        load_settings(config_path)


def test_load_settings_rejects_invalid_chunk_overlap(tmp_path):
    def mutate(raw):
        raw["chunking"]["chunk_size"] = 100
        raw["chunking"]["chunk_overlap"] = 100

    config_path = _write_config(tmp_path, mutate)

    with pytest.raises(ValueError, match="chunking.chunk_overlap"):
        load_settings(config_path)


def test_load_settings_rejects_missing_prompt_files(tmp_path):
    config_path = _write_config(
        tmp_path,
        lambda raw: raw["prompts"].update({"version": "missing-version"}),
    )

    with pytest.raises(ValueError, match="Prompt files missing"):
        load_settings(config_path)


def test_load_settings_rejects_store_paths_outside_project(tmp_path):
    config_path = _write_config(
        tmp_path,
        lambda raw: raw["store"].update({"bm25_index_path": "../bm25.pkl"}),
    )

    with pytest.raises(ValueError, match="store.bm25_index_path"):
        load_settings(config_path)
