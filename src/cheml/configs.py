from pathlib import Path
import os
from typing import Any


def get_settings_from_env(settings_name: str, default: Any = None) -> Any:
    return os.getenv(settings_name, default)  # type: ignore


BASE_PATH: Path = Path(
    get_settings_from_env("BASE_PATH", f"/tmp/cheml-{os.getpid()}")
)
