"""Shared helpers for the dictation-analysis toolkit.

Locates the app directory, resolves the OpenAI API key the same way the app
does, parses the transcript archive, and provides the German/English language
heuristic. Everything written by this toolkit goes to ``analysis/out/``.

This module is a library, but running it prints a short environment check::

    python analysis/common.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
TRANSCRIPTIONS_DIR = os.path.join(APP_DIR, "transcriptions")
OUT_DIR = os.path.join(HERE, "out")
LABELS_DIR = os.path.join(OUT_DIR, "labels")

CONFIG_FILE = os.path.join(APP_DIR, "whisper_config.json")
AI_INSIGHTS_FILE = os.path.join(APP_DIR, "ai_insights.json")

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Fixed activity vocabulary. Keep it generic and stable: the series and report
# scripts count these labels, so changing them invalidates existing labels.
ACTIVITIES = [
    "coding",
    "writing",
    "client work",
    "strategy",
    "research",
    "personal",
    "admin",
]


# --------------------------------------------------------------------------
# Language heuristic: reuse the app's word lists so this toolkit agrees with
# the in-app Speech Insights numbers.
# --------------------------------------------------------------------------

def _load_app_language_config():
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    try:
        import app_config  # type: ignore
        words = set(app_config.GERMAN_FUNCTION_WORDS)
        threshold = float(app_config.GERMAN_DETECT_THRESHOLD)
        return words, threshold
    except Exception:
        # Toolkit run outside the app directory: fall back to a small list so
        # the pipeline still works, just slightly less precisely.
        return (
            {
                "der", "die", "das", "und", "oder", "aber", "auch", "nicht",
                "ist", "sind", "war", "haben", "hat", "wird", "werden", "kann",
                "mit", "von", "aus", "auf", "im", "zum", "zur", "bei", "nach",
                "als", "wie", "wenn", "weil", "dass", "was", "wer", "wo",
                "ich", "mir", "mich", "wir", "uns", "sich", "dann", "doch",
                "sehr", "mehr", "für", "über", "unter", "ja", "also", "hier",
            },
            0.08,
        )


GERMAN_FUNCTION_WORDS, GERMAN_DETECT_THRESHOLD = _load_app_language_config()

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_GERMAN_UMLAUTS = set("äöüßÄÖÜ")


def tokenize_words(text: str):
    """Unicode-aware word tokenizer (keeps umlauts, drops digits/punctuation)."""
    return _WORD_RE.findall((text or "").lower())


def detect_language(text: str, tokens=None) -> str:
    """Cheap per-entry language heuristic -> 'de' or 'en'."""
    if not text:
        return "en"
    if any(ch in _GERMAN_UMLAUTS for ch in text):
        return "de"
    if tokens is None:
        tokens = tokenize_words(text)
    if not tokens:
        return "en"
    hits = sum(1 for t in tokens if t in GERMAN_FUNCTION_WORDS)
    return "de" if (hits / len(tokens)) >= GERMAN_DETECT_THRESHOLD else "en"


# --------------------------------------------------------------------------
# API key resolution (never printed, never written anywhere)
# --------------------------------------------------------------------------

def _hostname() -> str:
    return platform.node().replace(".", "_") or "machine"


def resolve_api_key() -> str:
    """Resolve the OpenAI key: OPENAI_API_KEY, else the app's config files.

    The app keeps the key in a per-machine overlay
    ``whisper_config.<HOSTNAME>.json`` and falls back to the shared
    ``whisper_config.json``. Returns "" when no key is configured.
    """
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    for path in (
        os.path.join(APP_DIR, f"whisper_config.{_hostname()}.json"),
        CONFIG_FILE,
    ):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        key = str(data.get("openai_api_key", "")).strip()
        if key:
            return key
    return ""


def require_api_key() -> str:
    key = resolve_api_key()
    if not key:
        raise SystemExit(
            "No OpenAI API key found. Set OPENAI_API_KEY, or configure the key "
            "in the app (Settings -> OpenAI API key)."
        )
    return key


def key_source() -> str:
    """Human-readable description of where the key came from (no secret)."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "OPENAI_API_KEY environment variable"
    local = os.path.join(APP_DIR, f"whisper_config.{_hostname()}.json")
    for path, label in ((local, "per-machine config overlay"),
                        (CONFIG_FILE, "whisper_config.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if str(json.load(fh).get("openai_api_key", "")).strip():
                    return label
        except Exception:
            continue
    return "not configured"


# --------------------------------------------------------------------------
# Transcript archive
# --------------------------------------------------------------------------

_LINE_TS_FMT = "%Y-%m-%d %H:%M:%S"


def transcript_files(directory: str | None = None):
    """Every transcript file in the archive, skipping sync-conflict copies."""
    directory = directory or TRANSCRIPTIONS_DIR
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".txt"):
            continue
        if "conflicted copy" in name.lower():
            continue  # sync debris would double-count entries
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            out.append(path)
    return out


def load_entries(directory: str | None = None, since: str | None = None,
                 until: str | None = None):
    """Parse the archive into ``[{"timestamp": datetime, "text": str}, ...]``.

    Archive format is one line per dictation: ``[YYYY-MM-DD HH:MM:SS] text``.
    """
    entries = []
    for path in transcript_files(directory):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line.startswith("[") or "] " not in line:
                        continue
                    ts_end = line.index("] ")
                    try:
                        ts = _dt.datetime.strptime(line[1:ts_end], _LINE_TS_FMT)
                    except ValueError:
                        continue
                    text = line[ts_end + 2:].strip()
                    if not text:
                        continue
                    day = ts.strftime("%Y-%m-%d")
                    if since and day < since:
                        continue
                    if until and day > until:
                        continue
                    entries.append({"timestamp": ts, "text": text})
        except Exception:
            continue
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def group_by_day(entries):
    """``{"YYYY-MM-DD": [entry, ...]}`` in chronological order."""
    days = defaultdict(list)
    for e in entries:
        days[e["timestamp"].strftime("%Y-%m-%d")].append(e)
    return dict(sorted(days.items()))


def day_text(day_entries, max_chars: int | None = None) -> str:
    text = " ".join(e["text"] for e in day_entries)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def day_german_share(day_entries) -> float:
    """Share of words in a day spoken in German, by the app's heuristic."""
    de = en = 0
    for e in day_entries:
        toks = tokenize_words(e["text"])
        if not toks:
            continue
        if detect_language(e["text"], toks) == "de":
            de += len(toks)
        else:
            en += len(toks)
    total = de + en
    return round(de / total, 4) if total else 0.0


def word_count(text: str) -> int:
    return len(tokenize_words(text))


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def ensure_out(*subdirs) -> str:
    path = os.path.join(OUT_DIR, *subdirs) if subdirs else OUT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def out_path(*parts) -> str:
    ensure_out(*parts[:-1])
    return os.path.join(OUT_DIR, *parts)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(path, data, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def load_labels(labels_dir: str | None = None):
    """Every cached day label as ``{"YYYY-MM-DD": {...}}``."""
    labels_dir = labels_dir or LABELS_DIR
    out = {}
    if not os.path.isdir(labels_dir):
        return out
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".json"):
            continue
        data = read_json(os.path.join(labels_dir, name))
        if isinstance(data, dict) and data.get("mood_score") is not None:
            out[name[:-5]] = data
    return out


def iso_week(day: str) -> str:
    y, w, _ = _dt.date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def week_start(week: str) -> str:
    return _dt.date.fromisocalendar(int(week[:4]), int(week[6:]), 1).isoformat()


def _main():
    import argparse
    argparse.ArgumentParser(
        description="Shared helpers. Run directly for an environment check: "
                    "archive size, date range, output directory and where the "
                    "OpenAI key is resolved from (never the key itself).",
    ).parse_args()
    entries = load_entries()
    days = group_by_day(entries)
    print(f"app directory     : {APP_DIR}")
    print(f"transcript files  : {len(transcript_files())}")
    print(f"entries parsed    : {len(entries):,}")
    print(f"days with speech  : {len(days)}")
    if days:
        first, last = min(days), max(days)
        print(f"range             : {first} .. {last}")
    print(f"output directory  : {OUT_DIR}")
    print(f"OpenAI key source : {key_source()}")


if __name__ == "__main__":
    _main()
