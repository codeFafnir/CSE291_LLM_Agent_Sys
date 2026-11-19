#!/usr/bin/env python3
"""Idempotent Harbor patch: every terminus-2 turn sends X-Session-ID.

Newer Harbor already supports ``session_id_headers``; this forces the default
to include ``X-Session-ID`` so Continuum can key pins by program. Older
installs without that knob get an ``extra_headers`` injection next to the
LiteLLM call kwargs. Do not fork Harbor; re-run after ``uv tool install``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

MARKER = "CSE291_CONTINUUM_X_SESSION_ID"
DEFAULT_OLD = 'session_id_headers: list[str] | None = Field(\n        default=None,'
DEFAULT_NEW = (
    'session_id_headers: list[str] | None = Field(\n'
    '        default=["X-Session-ID"],  # '
    + MARKER
    + "\n"
)
FALLBACK_NEEDLE = "extra_headers = self._llm_call_kwargs.setdefault(\"extra_headers\", {})"
FALLBACK_BLOCK = f'''# {MARKER}
        if "X-Session-ID" not in extra_headers:
            extra_headers["X-Session-ID"] = self._session_id
'''


def harbor_terminus_path() -> Path:
    spec = importlib.util.find_spec("harbor.agents.terminus_2.terminus_2")
    if spec is None or not spec.origin:
        raise SystemExit("harbor terminus_2 not installed; run uv tool install harbor first")
    return Path(spec.origin)


def patch(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        return "already patched"
    if "session_id_headers" in text and DEFAULT_OLD in text:
        path.write_text(text.replace(DEFAULT_OLD, DEFAULT_NEW, 1))
        return "defaulted session_id_headers to X-Session-ID"
    if FALLBACK_NEEDLE in text:
        path.write_text(text.replace(FALLBACK_NEEDLE, FALLBACK_NEEDLE + "\n" + FALLBACK_BLOCK, 1))
        return "injected X-Session-ID extra_headers fallback"
    # Last resort: force headers list right before the session_id_headers dict build.
    pattern = re.compile(
        r"(session_id_headers = \{\s*name: self\._session_id for name in self\._session_id_headers or \[\])",
        re.MULTILINE,
    )
    if pattern.search(text):
        replacement = (
            f"if not self._session_id_headers:  # {MARKER}\n"
            f'            self._session_id_headers = ["X-Session-ID"]\n'
            f"        \\1"
        )
        path.write_text(pattern.sub(replacement, text, count=1))
        return "forced empty session_id_headers to X-Session-ID"
    raise SystemExit(f"no known patch site in {path}")


def main() -> None:
    path = harbor_terminus_path()
    print(f"{patch(path)}: {path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        raise
