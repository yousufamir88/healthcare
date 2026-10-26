"""Tests for the Phase 0 foundation configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import (
    PROJECT_ROOT,
    apply_foundation_environment,
    get_settings,
    parse_bool,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_parse_bool_accepts_common_true_and_false_words():
    assert parse_bool("true", "MOCK_LLM") is True
    assert parse_bool("YES", "MOCK_LLM") is True
    assert parse_bool("0", "MOCK_LLM") is False
    assert parse_bool("off", "MOCK_LLM") is False


def test_parse_bool_rejects_unknown_text():
    with pytest.raises(ValueError, match="MOCK_LLM"):
        parse_bool("maybe", "MOCK_LLM")


def test_default_settings_use_mock_llm_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MOCK_LLM", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    settings = get_settings()

    assert settings.mock_llm is True
    assert settings.api_key_required is False
    assert settings.project_root == PROJECT_ROOT
    assert settings.knowledge_base_dir.is_dir()


def test_live_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")

    with pytest.raises(RuntimeError, match="MOCK_LLM must be true"):
        get_settings()


def test_invalid_mock_llm_value_is_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "maybe")

    with pytest.raises(ValueError, match="MOCK_LLM"):
        get_settings()


def test_dotenv_does_not_override_existing_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MOCK_LLM=false\nNEW_PHASE0_FLAG=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MOCK_LLM", "true")
    monkeypatch.delenv("NEW_PHASE0_FLAG", raising=False)

    try:
        apply_foundation_environment(env_file)

        assert os.environ["MOCK_LLM"] == "true"
        assert os.environ["NEW_PHASE0_FLAG"] == "from-file"
    finally:
        os.environ.pop("NEW_PHASE0_FLAG", None)


def test_clean_process_starts_without_api_keys_or_crewai():
    env = os.environ.copy()
    for key in (
        "MOCK_LLM",
        "CREWAI_DISABLE_TELEMETRY",
        "OTEL_SDK_DISABLED",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys\n"
            "import app.config\n"
            "from app.config import verify_foundation\n"
            "report = verify_foundation()\n"
            "assert report['mock_llm'] == 'true'\n"
            "assert report['api_key_required'] == 'false'\n"
            "assert report['status'] == 'ready'\n"
            "assert 'crewai' not in sys.modules\n"
            "print('clean-start-ok')\n"
        )],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "clean-start-ok" in result.stdout


def test_config_module_cli_prints_ready_status():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["MOCK_LLM"] = "true"

    result = subprocess.run(
        [sys.executable, "-m", "app.config"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "status: ready" in result.stdout
    assert "api_key_required: false" in result.stdout
