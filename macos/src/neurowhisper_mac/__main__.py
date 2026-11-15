"""Allow ``python -m neurowhisper_mac`` execution."""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
