"""Minimal .env loader (no python-dotenv dependency)."""
import os
from pathlib import Path


def load_dotenv(path=None):
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    path = Path(path) if path else Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_mly_api_key(explicit=None):
    """Return the Mapillary key from an explicit value or MLY_API_KEY in .env/environment."""
    if explicit:
        return explicit
    load_dotenv()
    key = os.environ.get("MLY_API_KEY")
    if not key:
        raise SystemExit("Mapillary API key missing: set MLY_API_KEY in .env or pass --api_key.")
    return key
