"""Central configuration for the Practo support agent.

Importing this module prepares a safe process environment:

- ``MOCK_LLM`` defaults to true, so graded demos do not call a paid model.
- CrewAI telemetry and the OpenTelemetry SDK default to disabled.
- No API key is read or required.

Later phases import ``get_settings()`` instead of scattering these decisions
through the rest of the code.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local env file.

    Existing environment variables win. A missing file is not an error:
    the defaults below are enough to start the project.
    """
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def apply_foundation_environment(env_file: Path | None = None) -> None:
    """Apply .env values, then fill any still-missing safe defaults."""
    _load_dotenv(ENV_FILE if env_file is None else env_file)
    os.environ.setdefault("MOCK_LLM", "true")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def parse_bool(value: str, name: str) -> bool:
    """Convert a configuration string into a boolean or raise ValueError."""
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be true or false. Received {value!r}."
    )


apply_foundation_environment()


class Settings(BaseModel):
    """Validated project settings. Defaults match the graded demonstration."""

    mock_llm: bool = True
    crewai_disable_telemetry: bool = True
    otel_sdk_disabled: bool = True
    project_name: str = "practo-domain-support-agent"
    project_root: Path = Field(default=PROJECT_ROOT)
    knowledge_base_dir: Path = Field(default=PROJECT_ROOT / "knowledge_base")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    api_key_required: bool = False

    def assert_graded_mode(self) -> None:
        """Stop startup when the process is not in the keyless demo mode."""
        if not self.mock_llm:
            raise RuntimeError(
                "MOCK_LLM must be true. This capstone does not call a paid "
                "model and does not require an API key."
            )
        if not self.crewai_disable_telemetry:
            raise RuntimeError(
                "CREWAI_DISABLE_TELEMETRY must be true before CrewAI runs."
            )
        if not self.otel_sdk_disabled:
            raise RuntimeError(
                "OTEL_SDK_DISABLED must be true before CrewAI runs."
            )
        if not self.knowledge_base_dir.is_dir():
            raise RuntimeError(
                "Knowledge base directory does not exist: "
                f"{self.knowledge_base_dir}"
            )


def load_settings() -> Settings:
    """Build Settings from the current environment."""
    try:
        settings = Settings(
            mock_llm=parse_bool(os.environ["MOCK_LLM"], "MOCK_LLM"),
            crewai_disable_telemetry=parse_bool(
                os.environ["CREWAI_DISABLE_TELEMETRY"],
                "CREWAI_DISABLE_TELEMETRY",
            ),
            otel_sdk_disabled=parse_bool(
                os.environ["OTEL_SDK_DISABLED"],
                "OTEL_SDK_DISABLED",
            ),
        )
    except ValidationError as exc:
        raise RuntimeError(f"Invalid project configuration: {exc}") from exc
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one shared Settings object for this process."""
    settings = load_settings()
    settings.assert_graded_mode()
    return settings


def verify_foundation() -> dict[str, str]:
    """Collect the startup report printed by ``python -m app.config``."""
    settings = get_settings()
    return {
        "project_name": settings.project_name,
        "mock_llm": str(settings.mock_llm).lower(),
        "api_key_required": str(settings.api_key_required).lower(),
        "crewai_disable_telemetry": os.environ["CREWAI_DISABLE_TELEMETRY"],
        "otel_sdk_disabled": os.environ["OTEL_SDK_DISABLED"],
        "knowledge_base_dir": str(settings.knowledge_base_dir),
        "status": "ready",
    }


def main() -> int:
    """Print the foundation report. Exit 1 when configuration is unsafe."""
    try:
        report = verify_foundation()
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    print("Practo support agent foundation")
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
