"""GitHub token: where it comes from, how it is stored, how it is shown.

Programs cannot log in to GitHub with a password (since 2021); they use a personal access
token. Looked for in this order, first match wins:
    1. environment variable PERFHOUND_GITHUB_TOKEN
    2. environment variable GITHUB_TOKEN
    3. the file written by `python -m perfhound.gateway login`:  ~/.perfhound/credentials.json
The file lives in the user's home folder - never in the project, so it cannot be committed.
The token is checked with GitHub BEFORE it is saved, is written atomically (temp file +
rename), and is only ever shown masked ("ghp_...5678"). Without a token GitHub allows 60
requests/hour on public repos; with one, 5,000.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VARS = ("PERFHOUND_GITHUB_TOKEN", "GITHUB_TOKEN")


def credentials_path() -> Path:
    base = os.environ.get("PERFHOUND_CACHE_DIR")
    return (Path(base) if base else Path.home() / ".perfhound") / "credentials.json"


def find_token() -> tuple[str | None, str | None]:
    """(token, where it came from) or (None, None)."""
    for var in ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value, f"environment variable {var}"
    path = credentials_path()
    try:
        token = json.loads(path.read_text(encoding="utf-8")).get("github_token", "").strip()
    except (OSError, ValueError, AttributeError):
        return None, None
    return (token, str(path)) if token else (None, None)


def save_token(token: str) -> Path:
    token = token.strip()
    if not token:
        raise ValueError("empty token")
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"github_token": token}), encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)                 # readable only by the user
    os.replace(tmp, path)                    # a crash never leaves half a file
    return path


def delete_token() -> bool:
    try:
        credentials_path().unlink()
        return True
    except FileNotFoundError:
        return False


def mask(token: str | None) -> str:
    if not token:
        return "(none)"
    return token[:4] + "..." + token[-4:] if len(token) > 12 else "***"
