"""Fast local lexical & pause longitudinal analytics engine for NeuroWhisper.

100% offline, deterministic, and zero-API:
- Computes length-invariant Moving-Average Type-Token Ratio (MATTR-50) per
  language (EN / DE) so Heap's Law and bilingual switching never distort trends.
- Extracts psycholinguistic self-tracking proxies for word-retrieval effort
  ("brain fog" placeholder/vague words), speech fluency (fillers, repetitions,
  clause restarts), burst-pause tail stalls (long pauses >=5s and P90 gap),
  forward VAD acoustic silence/WPM (when optional sidecar metadata exists),
  pronoun focus ("I/ich" vs "we/wir"), cognitive rigidity (absolutist markers),
  epistemic hedging, local vitality-vs-strain affect balance, and circadian
  late-night speech share.
- Maintains a compact per-machine incremental on-disk cache
  (``lexical_cache.<HOSTNAME>.json``) keyed by monthly ``(size, mtime_ns)``
  watermarks so warm loads and multi-year slicing complete in milliseconds
  without Dropbox sync conflicts.

Note: All indicators are personal longitudinal self-observation trends relative
to the speaker's own baseline, never clinical or medical diagnoses.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import platform
import re
import statistics
import uuid
from typing import Any

try:
    from app_config import (
        ACOUSTIC_META_SUBDIR,
        ENGLISH_FILLER_PATTERNS,
        ENGLISH_STOPWORDS,
        GERMAN_DETECT_THRESHOLD,
        GERMAN_FILLER_PATTERNS,
        GERMAN_FUNCTION_WORDS,
        GERMAN_STOPWORDS,
        INSIGHTS_MIN_WORDS_CONFIDENT,
        INSIGHTS_OUTLIER_SIGMA,
        INSIGHTS_RANGE_OPTIONS,
        LEXICAL_CACHE_PREFIX,
        PEER_BENCHMARKS,
        PERSONAL_BASELINE_DAYS,
        PERSONAL_BASELINE_MIN_DAYS,
        TRANSCRIPTIONS_DIR,
    )
except Exception:  # pragma: no cover - standalone fallback outside repo root
    ACOUSTIC_META_SUBDIR = "meta"
    LEXICAL_CACHE_PREFIX = "lexical_cache"
    TRANSCRIPTIONS_DIR = "transcriptions"
    PERSONAL_BASELINE_DAYS = 90
    PERSONAL_BASELINE_MIN_DAYS = 14
    INSIGHTS_MIN_WORDS_CONFIDENT = 40
    INSIGHTS_OUTLIER_SIGMA = 2.5
    GERMAN_DETECT_THRESHOLD = 0.08
    INSIGHTS_RANGE_OPTIONS = {
        "Today": 0,
        "7 days": 7,
        "30 days": 30,
        "90 days": 90,
        "6 months": 182,
        "This year": "ytd",
        "12 months": 365,
        "Prev 6 months": "prev_6m",
        "All": None,
    }
    PEER_BENCHMARKS = {}
    ENGLISH_FILLER_PATTERNS = {
        "um/uh": r"\b(um|uh)\b",
        "like": r"\blike\b",
        "you know": r"\byou know\b",
        "kind of": r"\bkind of\b",
        "sort of": r"\bsort of\b",
    }
    GERMAN_FILLER_PATTERNS = {
        "äh/ähm": r"\b(äh|ähm|öhm|öh)\b",
        "hm": r"\b(hm|hmm|mhm)\b",
        "halt/irgendwie": r"\b(halt|irgendwie)\b",
        "quasi/sozusagen": r"\b(quasi|sozusagen)\b",
        "sag ich mal": r"\b(sag ich mal|sage ich mal)\b",
    }
    ENGLISH_STOPWORDS = {
        "the", "and", "that", "this", "with", "from", "have", "just", "your", "you",
        "for", "are", "was", "were", "but", "not", "can", "could", "would", "should",
        "about", "into", "what", "when", "where", "which", "then", "than", "them",
        "they", "there", "their", "been", "being", "because", "also", "it's", "its",
        "i", "me", "my",
    }
    GERMAN_STOPWORDS = {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
        "und", "oder", "aber", "auch", "noch", "schon", "nur", "nicht", "ist",
        "sind", "war", "haben", "hat", "wird", "werden", "kann", "mit", "von",
        "aus", "auf", "im", "zum", "zur", "bei", "nach", "ich", "mir", "mich",
        "wir", "uns", "sich", "dann", "doch", "mal", "sehr", "mehr", "für", "über",
    }
    GERMAN_FUNCTION_WORDS = set(GERMAN_STOPWORDS)


CACHE_SCHEMA_VERSION = 2
MATTR_WINDOW = 50
BURST_GAP_MAX_SECONDS = 120.0

# 16 upper-bound edges (seconds) for O(1) slice quantile estimation (P90 pause)
PAUSE_HIST_EDGES = (
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0,
    10.0, 14.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0,
)

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_TRANSCRIPT_LINE_RX = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ?(.*)$")
_SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
_ABORT_CLAUSE_RX = re.compile(r"(?:\.\.\.|…|--|—)")
_GERMAN_UMLAUTS = frozenset("äöüßÄÖÜ")

_COMPILED_FILLERS = {
    "en": {k: re.compile(p) for k, p in ENGLISH_FILLER_PATTERNS.items()},
    "de": {k: re.compile(p) for k, p in GERMAN_FILLER_PATTERNS.items()},
}

# --- Curated Bilingual Psycholinguistic Lexicons (EN + DE) ---

ENGLISH_FUNCTION_CORE = frozenset({
    "the", "and", "that", "with", "from", "have", "this", "they", "would",
    "there", "their", "what", "about", "which", "when", "because",
})

VAGUE_WORDS = {
    "en": frozenset({
        "thing", "things", "stuff", "something", "anything", "somehow",
        "whatever", "whatsoever", "thingy", "whatchamacallit",
    }),
    "de": frozenset({
        "ding", "dinge", "dingen", "sache", "sachen", "zeug", "zeuge",
        "irgendwas", "irgendwie", "irgendwo", "irgendwann", "irgendwelche",
        "irgendwelchen", "irgendjemand", "dings", "dingsbums",
    }),
}

VAGUE_PHRASES_RX = {
    "en": re.compile(r"\b(?:kind of|sort of|kinda|sorta|whatnot)\b"),
    "de": re.compile(r"\b(?:oder so|und so weiter|sag ich mal|sage ich mal|wie heißt das)\b"),
}

PRONOUN_I = {
    "en": frozenset({"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself"}),
    "de": frozenset({
        "ich", "mir", "mich", "mein", "meine", "meinen", "meinem",
        "meiner", "meines", "meins", "selbst",
    }),
}

PRONOUN_WE = {
    "en": frozenset({"we", "we're", "we've", "we'd", "we'll", "us", "our", "ours", "ourselves"}),
    "de": frozenset({
        "wir", "uns", "unser", "unsere", "unseren", "unserem", "unserer", "unseres",
    }),
}

ABSOLUTIST_WORDS = {
    "en": frozenset({
        "absolutely", "all", "always", "complete", "completely", "constant",
        "constantly", "definitely", "entire", "entirely", "ever", "every",
        "everyone", "everything", "full", "must", "never", "nothing",
        "totally", "whole", "impossible",
    }),
    "de": frozenset({
        "absolut", "alle", "alles", "immer", "komplett", "ständig",
        "definitiv", "völlig", "jeder", "jede", "jedes", "jedermann",
        "muss", "müssen", "nie", "niemals", "nichts", "total", "unmöglich", "ewig",
    }),
}

HEDGE_WORDS = {
    "en": frozenset({
        "maybe", "perhaps", "probably", "possibly", "guess", "suppose",
        "unsure", "unclear", "confused", "apparently", "roughly",
    }),
    "de": frozenset({
        "vielleicht", "wahrscheinlich", "vermutlich", "möglicherweise",
        "eventuell", "unklar", "unsicher", "verwirrt", "angeblich", "ungefähr",
    }),
}

HEDGE_PHRASES_RX = {
    "en": re.compile(r"\b(?:not sure|don't know|do not know|no idea|i think)\b"),
    "de": re.compile(r"\b(?:keine ahnung|weiß nicht|weiss nicht|nicht sicher|ich glaube)\b"),
}

VITALITY_POS_WORDS = {
    "en": frozenset({
        "good", "great", "clear", "focused", "energized", "productive",
        "happy", "excited", "calm", "better", "progress", "solved",
        "ready", "strong", "confident", "sharp", "grateful", "smooth", "easy",
    }),
    "de": frozenset({
        "gut", "super", "klar", "fokussiert", "konzentriert", "produktiv",
        "froh", "glücklich", "ruhig", "besser", "fortschritt", "gelöst",
        "bereit", "stark", "dankbar", "fit", "wach", "entspannt", "einfach",
    }),
}

STRAIN_FATIGUE_WORDS = {
    "en": frozenset({
        "tired", "exhausted", "fatigued", "foggy", "drained", "overwhelmed",
        "stressed", "anxious", "frustrated", "stuck", "slow", "headache",
        "burnout", "burned", "sleepy", "scattered", "forgetful", "annoyed",
        "worried", "exhausting", "dizzy", "sick",
    }),
    "de": frozenset({
        "müde", "erschöpft", "kaputt", "neblig", "ausgelaugt", "überfordert",
        "gestresst", "ängstlich", "frustriert", "blockiert", "langsam",
        "kopfschmerzen", "burnout", "schläfrig", "zerstreut", "vergesslich",
        "genervt", "besorgt", "anstrengend", "krank", "schwindelig",
    }),
}


# --- Metric Metadata Registry ---

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "clarity_index": {
        "label": "Cognitive Clarity Index",
        "short": "Clarity Index",
        "unit": "",
        "decimals": 1,
        "good_direction": "higher",
        "two_sided_outlier": True,
        "color": "#2dd4bf",
        "peer_key": "clarity_index",
        "description": (
            "Personal composite (50 = your 90-day norm) combining vocabulary variety (+MATTR-50), "
            "word specificity (-vague placeholders), verbal fluency (-fillers & self-repairs), "
            "and continuity (-long pauses >=5s)."
        ),
    },
    "mattr_diversity": {
        "label": "Lexical Diversity (MATTR-50)",
        "short": "MATTR-50",
        "unit": "%",
        "decimals": 1,
        "good_direction": "higher",
        "two_sided_outlier": True,
        "color": "#34d399",
        "peer_key": "mattr_diversity",
        "description": (
            "Moving-Average Type-Token Ratio: distinct words per 50-word sliding window (EN/DE aligned). "
            "Unlike raw unique-word %, MATTR-50 stays fair across short days and multi-month horizons."
        ),
    },
    "vague_per_100": {
        "label": "Word-Finding Load (Vague Words)",
        "short": "Vague / 100w",
        "unit": "/100w",
        "decimals": 2,
        "good_direction": "lower",
        "color": "#f59e0b",
        "peer_key": "vague_per_100",
        "description": (
            "Generic placeholder words ('thing, stuff, something, sort of / Ding, Zeug, irgendwas') "
            "per 100 words — tracks how often general stand-ins replace precise nouns."
        ),
    },
    "pause_long_pct": {
        "label": "Long Pauses (>=5s Share)",
        "short": "Long Pauses %",
        "unit": "%",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#ef4444",
        "peer_key": "pause_long_pct",
        "description": (
            "Percentage of within-session dictation gaps (<=120s) lasting 5 seconds or longer — "
            "captures mid-chain stalls, retrieval hesitations, or thinking pauses between utterances."
        ),
    },
    "pause_p90_s": {
        "label": "90th-Percentile Pause (Tail Stall)",
        "short": "P90 Pause",
        "unit": "s",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#fb7185",
        "peer_key": "pause_p90_s",
        "description": (
            "Duration (in seconds) of your longest 10% of within-burst pauses — sensitive to "
            "occasional heavy mid-thought blocks that average pause time can hide."
        ),
    },
    "pause_avg_s": {
        "label": "Average Burst Pause",
        "short": "Avg Pause",
        "unit": "s",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#f97316",
        "peer_key": "pause_avg_seconds",
        "description": (
            "Mean timestamp gap (up to 120s) between consecutive utterances in an active dictation burst."
        ),
    },
    "fillers_per_100": {
        "label": "Filler Words",
        "short": "Fillers / 100w",
        "unit": "/100w",
        "decimals": 2,
        "good_direction": "lower",
        "color": "#38bdf8",
        "peer_key": "clarity_fillers_per_100",
        "description": (
            "Vocalized hesitation markers ('um, uh, like, you know / äh, ähm, halt, quasi') per 100 words — "
            "reflects real-time speech planning load while formulating thoughts."
        ),
    },
    "disfluency_per_1k": {
        "label": "Repetitions & Self-Repairs",
        "short": "Repairs / 1k",
        "unit": "/1k",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#818cf8",
        "peer_key": "clarity_repetition_per_1k",
        "description": (
            "Immediate word/phrase repetitions ('the the', 'I think I think') plus mid-sentence restarts "
            "('...', '--') per 1,000 words — tracks working-memory restarts during speech."
        ),
    },
    "content_density_pct": {
        "label": "Content-Word Density",
        "short": "Content %",
        "unit": "%",
        "decimals": 1,
        "good_direction": "higher",
        "two_sided_outlier": True,
        "color": "#22d3ee",
        "peer_key": "content_density_pct",
        "description": (
            "Share of substantive content words (nouns, verbs, adjectives) vs. grammatical glue words, "
            "normalized across English and German — a classic proxy for propositional idea density."
        ),
    },
    "lexical_long_word_percent": {
        "label": "Vocabulary Depth (7+ Char Words)",
        "short": "7+ Chars %",
        "unit": "%",
        "decimals": 1,
        "good_direction": "higher",
        "two_sided_outlier": True,
        "color": "#0ea5e9",
        "peer_key": "lexical_long_word_percent",
        "description": (
            "Percentage of words with 7 or more characters (EN/DE offset-adjusted) — reflects technical "
            "precision and morphological complexity vs. simplified everyday phrasing."
        ),
    },
    "pace_words_entry": {
        "label": "Sustained Utterance Length",
        "short": "Words / Entry",
        "unit": " w",
        "decimals": 1,
        "good_direction": "higher",
        "color": "#2dd4bf",
        "peer_key": "pace_avg_words_entry",
        "description": (
            "Average words spoken per dictation chunk — distinguishes sustained multi-clause reasoning "
            "from fragmented bursts or quick one-line commands."
        ),
    },
    "words_per_sentence": {
        "label": "Mean Sentence Length",
        "short": "Words / Sent",
        "unit": " w",
        "decimals": 1,
        "good_direction": "neutral",
        "color": "#60a5fa",
        "peer_key": "pace_avg_words_entry",
        "description": (
            "Average words per punctuated sentence — tracks syntactic elaboration and clause complexity."
        ),
    },
    "self_focus_per_100": {
        "label": "Self-Focus ('I / ich')",
        "short": "I-Talk / 100w",
        "unit": "/100w",
        "decimals": 2,
        "good_direction": "neutral",
        "color": "#c084fc",
        "peer_key": "self_focus_per_100",
        "description": (
            "1st-person singular pronouns ('I, me, my / ich, mir, mich, mein*') per 100 words — tracks "
            "inward self-referential focus vs. external/task-focused or collective ('we/wir') framing."
        ),
    },
    "absolutist_per_100": {
        "label": "Cognitive Rigidity (Absolutist)",
        "short": "Absolutist / 100w",
        "unit": "/100w",
        "decimals": 2,
        "good_direction": "lower",
        "color": "#f472b6",
        "peer_key": "absolutist_per_100",
        "description": (
            "All-or-nothing words ('always, never, completely, nothing / immer, nie, völlig, nichts') "
            "per 100 words — tracks categorical framing vs. nuanced qualification."
        ),
    },
    "hedges_per_100": {
        "label": "Epistemic Uncertainty (Hedges)",
        "short": "Hedges / 100w",
        "unit": "/100w",
        "decimals": 2,
        "good_direction": "lower",
        "color": "#fbbf24",
        "peer_key": "hedges_per_100",
        "description": (
            "Tentative qualifiers ('maybe, perhaps, not sure / vielleicht, wahrscheinlich, keine Ahnung') "
            "per 100 words — rises during exploratory brainstorming or low confidence."
        ),
    },
    "vitality_balance": {
        "label": "Lexical Vitality vs. Strain",
        "short": "Vitality Score",
        "unit": "",
        "decimals": 1,
        "good_direction": "higher",
        "two_sided_outlier": True,
        "color": "#4ade80",
        "peer_key": "vitality_balance",
        "description": (
            "100% offline affect word balance centered at 50 (>50 = more energy/clarity/progress words; "
            "<50 = more tired/foggy/overwhelmed/stressed words)."
        ),
    },
    "late_night_pct": {
        "label": "Late-Night Speech (22:00–05:00)",
        "short": "Late-Night %",
        "unit": "%",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#a78bfa",
        "peer_key": "late_night_pct",
        "description": (
            "Percentage of your dictated words spoken between 22:00 and 04:59 — a local behavioral "
            "proxy for evening work extension and sleep-window encroachment."
        ),
    },
    "acoustic_wpm": {
        "label": "Articulation Speed (Acoustic WPM)",
        "short": "Acoustic WPM",
        "unit": " WPM",
        "decimals": 0,
        "good_direction": "neutral",
        "color": "#14b8a6",
        "peer_key": "acoustic_wpm",
        "description": (
            "True speaking rate (words per minute of active voice, excluding silence) captured from "
            "local ASR segment timestamps on new recordings."
        ),
    },
    "acoustic_silence_pct": {
        "label": "Within-Utterance Silence (VAD)",
        "short": "VAD Silence %",
        "unit": "%",
        "decimals": 1,
        "good_direction": "lower",
        "color": "#fb923c",
        "peer_key": "acoustic_silence_pct",
        "description": (
            "Share of non-speech silence inside individual recordings (captured forward via local ASR "
            "segment boundaries) — measures mid-sentence hesitation or open-mic dead air."
        ),
    },
    "volume_words": {
        "label": "Dictated Volume",
        "short": "Words",
        "unit": " w",
        "decimals": 0,
        "good_direction": "higher",
        "color": "#38bdf8",
        "peer_key": None,
        "description": (
            "Total words dictated per time bucket — provides context so rate changes on quiet days "
            "aren't over-interpreted."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pure Lexical & Timing Primitives
# ---------------------------------------------------------------------------

def default_hostname() -> str:
    return platform.node().replace(".", "_") or "machine"


def tokenize_words(text: str) -> list[str]:
    """Unicode-aware word tokenizer (preserves umlauts/apostrophes, drops digits)."""
    return _WORD_RE.findall((text or "").lower())


def detect_entry_language(text: str, tokens: list[str] | None = None) -> str:
    """Fast deterministic language classifier -> 'de' or 'en'."""
    if not text:
        return "en"
    if tokens is None:
        tokens = tokenize_words(text)
    umlaut_count = sum(1 for ch in text if ch in _GERMAN_UMLAUTS)
    if not tokens:
        return "de" if umlaut_count > 0 else "en"
    hits = sum(1 for t in tokens if t in GERMAN_FUNCTION_WORDS)
    if umlaut_count > 0 and (umlaut_count >= 2 or len(tokens) <= 5 or hits >= 1):
        return "de"
    return "de" if (hits / len(tokens)) >= GERMAN_DETECT_THRESHOLD else "en"


def compute_mattr(tokens: list[str], window: int = MATTR_WINDOW) -> tuple[float, float]:
    """Exact O(N) Moving-Average Type-Token Ratio sufficient statistics.

    Returns ``(weighted_ratio_sum, weight_windows)`` where each ratio is in ``[0, 1]``:
    - For ``len(tokens) >= window``: exact sliding-window average across all
      ``N - window + 1`` contiguous windows, weighted by window count.
    - For ``8 <= len(tokens) < window``: Heap's-law calibrated projection onto
      ``window`` with reduced fractional weight so short days do not spike upward
      yet still contribute gently when viewed individually.
    - For ``len(tokens) < 8``: ``(0.0, 0.0)`` (insufficient sample for diversity).
    """
    n = len(tokens)
    if n < 8:
        return 0.0, 0.0
    if n < window:
        raw_ttr = len(set(tokens)) / float(n)
        # Under Heap's Law TTR(W) / TTR(N) ~= (N / W)^0.22 for short speech runs
        projected = max(0.15, min(0.96, raw_ttr * ((n / float(window)) ** 0.22)))
        weight = (n / float(window)) * 0.5
        return projected * weight, weight

    freq: dict[str, int] = {}
    unique = 0
    sum_unique = 0
    for i, tok in enumerate(tokens):
        prev = freq.get(tok, 0)
        if prev == 0:
            unique += 1
        freq[tok] = prev + 1
        if i >= window:
            old = tokens[i - window]
            rem = freq[old] - 1
            freq[old] = rem
            if rem == 0:
                unique -= 1
        if i >= window - 1:
            sum_unique += unique

    num_windows = float(n - window + 1)
    mean_ratio = sum_unique / (num_windows * float(window))
    return mean_ratio * num_windows, num_windows


def _pause_bin_index(gap_s: float) -> int:
    for idx, edge in enumerate(PAUSE_HIST_EDGES):
        if gap_s <= edge:
            return idx
    return len(PAUSE_HIST_EDGES) - 1


def quantile_from_pause_hist(hist: list[int], total_count: int, q: float = 0.90) -> float:
    """Interpolate quantile (e.g. P90) in O(1) from the 16-bin pause histogram."""
    if total_count <= 0 or not hist:
        return 0.0
    clean_hist = [max(0, int(c or 0)) for c in hist]
    eff_total = sum(clean_hist)
    if eff_total <= 0:
        return 0.0
    target = max(0.0, min(float(eff_total), eff_total * max(0.0, min(1.0, float(q)))))
    if target <= 0.0:
        return 0.0
    cum = 0
    prev_edge = 0.0
    for idx, count in enumerate(clean_hist):
        edge = PAUSE_HIST_EDGES[idx] if idx < len(PAUSE_HIST_EDGES) else BURST_GAP_MAX_SECONDS
        if cum + count >= target and count > 0:
            frac = (target - cum) / float(count)
            return prev_edge + max(0.0, min(1.0, frac)) * (edge - prev_edge)
        cum += count
        prev_edge = edge
    return PAUSE_HIST_EDGES[-1]


def extract_acoustic_meta(audio_s: float, segments: Any = None, words_count: int | None = None) -> dict[str, Any]:
    """Extract lightweight local timing & internal pause stats from ASR segments.

    Works with ``faster-whisper`` Segment namedtuples, ``onnx-asr`` objects, or
    dicts exposing ``start``/``end`` timestamps. Never retains text or audio.
    """
    try:
        raw_audio_s = float(audio_s or 0.0)
    except (TypeError, ValueError):
        raw_audio_s = 0.0
    audio_s = max(0.0, raw_audio_s) if math.isfinite(raw_audio_s) else 0.0
    meta: dict[str, Any] = {"audio_s": round(audio_s, 2)}
    if words_count is not None:
        meta["words"] = max(0, int(words_count))

    if segments is None:
        return meta

    spans: list[tuple[float, float]] = []
    seg_texts: list[str] = []
    try:
        for seg in segments:
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            txt = getattr(seg, "text", None)
            if isinstance(seg, dict):
                start = seg.get("start", start)
                end = seg.get("end", end)
                txt = seg.get("text", txt)
            if txt:
                seg_texts.append(str(txt))
            if start is not None and end is not None:
                s_f, e_f = float(start), float(end)
                if math.isfinite(s_f) and math.isfinite(e_f) and e_f > s_f >= 0.0:
                    spans.append((s_f, e_f))
    except Exception:
        return meta

    if words_count is None and seg_texts:
        meta["words"] = len(tokenize_words(" ".join(seg_texts)))

    if spans:
        spans.sort(key=lambda x: x[0])
        merged: list[tuple[float, float]] = [spans[0]]
        for s_f, e_f in spans[1:]:
            prev_s, prev_e = merged[-1]
            if s_f <= prev_e:
                merged[-1] = (prev_s, max(prev_e, e_f))
            else:
                merged.append((s_f, e_f))

        speech_s = sum(e - s for s, e in merged)
        if audio_s > 0.0:
            speech_s = min(audio_s, speech_s)
        internal_pauses = 0
        long_pauses = 0
        max_pause_s = 0.0
        for i in range(1, len(merged)):
            gap = merged[i][0] - merged[i - 1][1]
            if gap >= 0.45:
                internal_pauses += 1
                if gap >= 1.5:
                    long_pauses += 1
                if gap > max_pause_s:
                    max_pause_s = gap
        lead_gap = merged[0][0]
        if lead_gap >= 1.5 and lead_gap > max_pause_s:
            max_pause_s = lead_gap
        if audio_s > merged[-1][1]:
            tail_gap = audio_s - merged[-1][1]
            if tail_gap >= 1.5 and tail_gap > max_pause_s:
                max_pause_s = tail_gap
        meta["speech_s"] = round(speech_s, 2)
        meta["pauses"] = internal_pauses
        meta["long_pauses"] = long_pauses
        meta["max_pause_s"] = round(max_pause_s, 2)
    return meta


def merge_batch_acoustic_metas(metas: list[dict[str, Any]], total_words: int | None = None) -> dict[str, Any]:
    """Combine per-chunk acoustic metadata from a multi-chunk Batch session."""
    valid = [
        m for m in metas
        if isinstance(m, dict)
        and isinstance(m.get("audio_s"), (int, float))
        and math.isfinite(float(m["audio_s"]))
        and float(m["audio_s"]) > 0.0
    ]
    if not valid:
        return {}
    audio_s = sum(float(m["audio_s"]) for m in valid)
    with_speech = [
        m for m in valid
        if "speech_s" in m and m["speech_s"] is not None and math.isfinite(float(m["speech_s"]))
    ]
    out: dict[str, Any] = {"audio_s": round(audio_s, 2)}
    if total_words is not None:
        out["words"] = max(0, int(total_words))
    else:
        out["words"] = sum(max(0, int(m.get("words", 0) or 0)) for m in valid)
    if with_speech:
        speech_s = sum(max(0.0, float(m["speech_s"])) for m in with_speech)
        eff_audio = sum(float(m["audio_s"]) for m in with_speech)
        out["audio_s"] = round(eff_audio, 2)
        out["speech_s"] = round(min(eff_audio, speech_s), 2)
        out["pauses"] = sum(max(0, int(m.get("pauses", 0) or 0)) for m in with_speech)
        out["long_pauses"] = sum(max(0, int(m.get("long_pauses", 0) or 0)) for m in with_speech)
        out["max_pause_s"] = round(max(float(m.get("max_pause_s", 0.0) or 0.0) for m in with_speech), 2)
    return out


def append_acoustic_meta(
    transcriptions_dir: str,
    hostname: str,
    timestamp: dt.datetime,
    meta: dict[str, Any] | None,
) -> None:
    """Append one compact per-utterance timing record to ``transcriptions/meta/YYYY-MM.HOST.jsonl``."""
    if not meta or float(meta.get("audio_s", 0.0)) <= 0.0:
        return
    meta_dir = os.path.join(transcriptions_dir, ACOUSTIC_META_SUBDIR)
    os.makedirs(meta_dir, exist_ok=True)
    filename = f"{timestamp.strftime('%Y-%m')}.{hostname}.jsonl"
    path = os.path.join(meta_dir, filename)
    record = {"ts": timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    for key in ("audio_s", "speech_s", "pauses", "long_pauses", "max_pause_s", "words"):
        if key in meta and meta[key] is not None:
            record[key] = meta[key]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Daily Sufficient Statistics Builder
# ---------------------------------------------------------------------------

def _empty_lang_bucket() -> dict[str, float | int]:
    return {
        "entries": 0,
        "words": 0,
        "sentences": 0,
        "long_words": 0,
        "stop_words": 0,
        "fillers": 0,
        "repeats": 0,
        "aborts": 0,
        "vague": 0,
        "pronoun_i": 0,
        "pronoun_we": 0,
        "absolutist": 0,
        "hedges": 0,
        "pos_vitality": 0,
        "neg_strain": 0,
        "codeswitch_hits": 0,
        "mattr_sum": 0.0,
        "mattr_w": 0.0,
    }


def _empty_day_record() -> dict[str, Any]:
    return {
        "entries": 0,
        "words": 0,
        "en": _empty_lang_bucket(),
        "de": _empty_lang_bucket(),
        "gap_count": 0,
        "gap_sum_s": 0.0,
        "gap_short": 0,
        "gap_med": 0,
        "gap_long": 0,
        "gap_vlong": 0,
        "gap_hist": [0] * len(PAUSE_HIST_EDGES),
        "hours": [0] * 24,
        "ac_audio_s": 0.0,
        "ac_speech_s": 0.0,
        "ac_words": 0,
        "ac_pauses": 0,
        "ac_long_pauses": 0,
    }


def _count_entry_features(text: str, tokens: list[str], lang: str) -> dict[str, int]:
    """Single-pass lexical feature extraction for one transcript entry."""
    lowered = text.lower()
    stop_set = GERMAN_STOPWORDS if lang == "de" else ENGLISH_STOPWORDS
    vague_set = VAGUE_WORDS[lang]
    i_set = PRONOUN_I[lang]
    we_set = PRONOUN_WE[lang]
    abs_set = ABSOLUTIST_WORDS[lang]
    hedge_set = HEDGE_WORDS[lang]
    pos_set = VITALITY_POS_WORDS[lang]
    neg_set = STRAIN_FATIGUE_WORDS[lang]
    other_func = ENGLISH_FUNCTION_CORE if lang == "de" else GERMAN_FUNCTION_WORDS

    long_words = 0
    stop_words = 0
    vague = 0
    pronoun_i = 0
    pronoun_we = 0
    absolutist = 0
    hedges = 0
    pos_vitality = 0
    neg_strain = 0
    codeswitch_hits = 0
    repeats = 0

    n = len(tokens)
    for idx, tok in enumerate(tokens):
        if len(tok) >= 7:
            long_words += 1
        if tok in stop_set:
            stop_words += 1
        if tok in vague_set:
            vague += 1
        if tok in i_set:
            pronoun_i += 1
        elif tok in we_set:
            pronoun_we += 1
        if tok in abs_set:
            absolutist += 1
        if tok in hedge_set:
            hedges += 1
        if tok in pos_set:
            pos_vitality += 1
        elif tok in neg_set:
            neg_strain += 1
        if n >= 8 and tok in other_func:
            codeswitch_hits += 1
        if idx > 0 and tok == tokens[idx - 1]:
            repeats += 1
        elif idx >= 3 and tok == tokens[idx - 2] and tokens[idx - 1] == tokens[idx - 3]:
            repeats += 1

    # Multi-word phrases & vocalized fillers
    fillers = sum(len(rx.findall(lowered)) for rx in _COMPILED_FILLERS[lang].values())
    vague += len(VAGUE_PHRASES_RX[lang].findall(lowered))
    hedges += len(HEDGE_PHRASES_RX[lang].findall(lowered))
    aborts = len(_ABORT_CLAUSE_RX.findall(text))
    sents = [s for s in _SENTENCE_SPLIT_RX.split(text.strip()) if s.strip()]
    sentences = max(1, len(sents)) if n > 0 else 0

    return {
        "sentences": sentences,
        "long_words": long_words,
        "stop_words": stop_words,
        "fillers": fillers,
        "repeats": repeats,
        "aborts": aborts,
        "vague": vague,
        "pronoun_i": pronoun_i,
        "pronoun_we": pronoun_we,
        "absolutist": absolutist,
        "hedges": hedges,
        "pos_vitality": pos_vitality,
        "neg_strain": neg_strain,
        "codeswitch_hits": codeswitch_hits,
    }


def _fold_day_entries(
    day_entries: list[dict[str, Any]],
    prev_ts: dt.datetime | None = None,
    day_acoustic: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dt.datetime | None]:
    """Build one compact daily record from chronologically ordered entries of a day."""
    rec = _empty_day_record()
    lang_tokens: dict[str, list[str]] = {"en": [], "de": []}

    for entry in day_entries:
        ts: dt.datetime = entry["timestamp"]
        text: str = entry["text"]
        toks = tokenize_words(text)
        n = len(toks)
        lang = detect_entry_language(text, toks)

        rec["entries"] += 1
        rec["words"] += n
        rec["hours"][ts.hour] += n

        if prev_ts is not None:
            gap = (ts - prev_ts).total_seconds()
            if 0.0 < gap <= BURST_GAP_MAX_SECONDS:
                rec["gap_count"] += 1
                rec["gap_sum_s"] = round(rec["gap_sum_s"] + gap, 3)
                if gap < 2.0:
                    rec["gap_short"] += 1
                elif gap < 5.0:
                    rec["gap_med"] += 1
                else:
                    rec["gap_long"] += 1
                    if gap >= 10.0:
                        rec["gap_vlong"] += 1
                rec["gap_hist"][_pause_bin_index(gap)] += 1
        prev_ts = ts

        if n == 0:
            continue

        lang_tokens[lang].extend(toks)
        feats = _count_entry_features(text, toks, lang)
        lb = rec[lang]
        lb["entries"] += 1
        lb["words"] += n
        for fk, fv in feats.items():
            lb[fk] += fv

    for lang in ("en", "de"):
        m_sum, m_w = compute_mattr(lang_tokens[lang], MATTR_WINDOW)
        rec[lang]["mattr_sum"] = round(m_sum, 4)
        rec[lang]["mattr_w"] = round(m_w, 4)

    if day_acoustic:
        for ac in day_acoustic:
            _accumulate_acoustic_entry(rec, ac)

    return rec, prev_ts


def _accumulate_acoustic_entry(rec: dict[str, Any], ac: Any) -> None:
    """Safely accumulate validated VAD acoustic sidecar fields into ``rec``."""
    if not isinstance(ac, dict):
        return
    if "speech_s" not in ac or ac["speech_s"] is None:
        return
    try:
        a_s = float(ac.get("audio_s", 0.0) or 0.0)
        s_s = float(ac["speech_s"])
        words = int(ac.get("words", 0) or 0)
        pauses = int(ac.get("pauses", 0) or 0)
        long_pauses = int(ac.get("long_pauses", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return
    if not (math.isfinite(a_s) and math.isfinite(s_s)) or a_s <= 0.0:
        return
    s_clamped = max(0.0, min(a_s, s_s))
    rec["ac_audio_s"] = round(rec["ac_audio_s"] + a_s, 2)
    rec["ac_speech_s"] = round(rec["ac_speech_s"] + s_clamped, 2)
    rec["ac_words"] += max(0, words)
    rec["ac_pauses"] += max(0, pauses)
    rec["ac_long_pauses"] += max(0, long_pauses)


def _merge_day_records(target: dict[str, Any], src: dict[str, Any]) -> None:
    """Add ``src`` day sufficient statistics into ``target`` in-place."""
    if not isinstance(src, dict):
        return
    target["entries"] += int(src.get("entries", 0) or 0)
    target["words"] += int(src.get("words", 0) or 0)
    for lang in ("en", "de"):
        s_lb = src.get(lang)
        if not isinstance(s_lb, dict):
            continue
        t_lb = target[lang]
        for k in t_lb:
            if k in ("mattr_sum", "mattr_w"):
                val = float(s_lb.get(k, 0.0) or 0.0)
                if math.isfinite(val):
                    t_lb[k] = round(float(t_lb[k]) + val, 4)
            else:
                t_lb[k] = int(t_lb[k]) + int(s_lb.get(k, 0) or 0)
    target["gap_count"] += int(src.get("gap_count", 0) or 0)
    gap_s_val = float(src.get("gap_sum_s", 0.0) or 0.0)
    if math.isfinite(gap_s_val):
        target["gap_sum_s"] = round(target["gap_sum_s"] + gap_s_val, 3)
    target["gap_short"] += int(src.get("gap_short", 0) or 0)
    target["gap_med"] += int(src.get("gap_med", 0) or 0)
    target["gap_long"] += int(src.get("gap_long", 0) or 0)
    target["gap_vlong"] += int(src.get("gap_vlong", 0) or 0)
    src_hist = src.get("gap_hist") or []
    if isinstance(src_hist, list):
        for i in range(min(len(target["gap_hist"]), len(src_hist))):
            target["gap_hist"][i] += int(src_hist[i] or 0)
    src_hours = src.get("hours") or []
    if isinstance(src_hours, list):
        for h in range(min(24, len(src_hours))):
            target["hours"][h] += int(src_hours[h] or 0)
    ac_a = float(src.get("ac_audio_s", 0.0) or 0.0)
    ac_s = float(src.get("ac_speech_s", 0.0) or 0.0)
    if math.isfinite(ac_a) and math.isfinite(ac_s):
        target["ac_audio_s"] = round(target["ac_audio_s"] + ac_a, 2)
        target["ac_speech_s"] = round(target["ac_speech_s"] + ac_s, 2)
    target["ac_words"] += int(src.get("ac_words", 0) or 0)
    target["ac_pauses"] += int(src.get("ac_pauses", 0) or 0)
    target["ac_long_pauses"] += int(src.get("ac_long_pauses", 0) or 0)


# ---------------------------------------------------------------------------
# Transcript & Metadata File Parsing
# ---------------------------------------------------------------------------

def parse_transcript_file(filepath: str) -> list[dict[str, Any]]:
    """Parse ``[YYYY-MM-DD HH:MM:SS] text`` (including multi-line entries)."""
    entries: list[dict[str, Any]] = []
    current_ts: dt.datetime | None = None
    current_lines: list[str] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                stripped = raw_line.rstrip("\r\n")
                m = _TRANSCRIPT_LINE_RX.match(stripped)
                if m:
                    if current_ts is not None:
                        txt = "\n".join(current_lines).strip()
                        if txt:
                            entries.append({"timestamp": current_ts, "text": txt})
                    try:
                        current_ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        current_lines = [m.group(2)]
                    except ValueError:
                        current_ts = None
                        current_lines = []
                elif current_ts is not None:
                    current_lines.append(stripped)
        if current_ts is not None:
            txt = "\n".join(current_lines).strip()
            if txt:
                entries.append({"timestamp": current_ts, "text": txt})
    except OSError:
        return []
    return entries


def parse_acoustic_meta_file(filepath: str) -> dict[str, list[dict[str, Any]]]:
    """Parse ``YYYY-MM.HOST.jsonl`` into ``{day_iso: [meta_dict, ...]}``."""
    by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                ts_str = str(rec.get("ts", ""))
                if len(ts_str) >= 10:
                    day_iso = ts_str[:10]
                    try:
                        dt.date.fromisoformat(day_iso)
                    except ValueError:
                        continue
                    by_day[day_iso].append(rec)
    except OSError:
        pass
    return dict(by_day)


def _month_bucket_key(filename: str) -> str:
    if len(filename) >= 7 and filename[:4].isdigit() and filename[4] == "-" and filename[5:7].isdigit():
        return filename[:7]
    return "misc"


# ---------------------------------------------------------------------------
# Incremental Per-Machine On-Disk Lexical Index
# ---------------------------------------------------------------------------

class LexicalDiskIndex:
    """Incremental, Dropbox-safe daily lexical & timing index.

    Persists to ``lexical_cache.<HOSTNAME>.json`` (never shared across machines,
    avoiding sync conflicts while indexing all machines' transcript files).
    """

    def __init__(
        self,
        transcriptions_dir: str = TRANSCRIPTIONS_DIR,
        cache_path: str | None = None,
        hostname: str | None = None,
    ) -> None:
        self.transcriptions_dir = transcriptions_dir
        self.hostname = hostname or default_hostname()
        if cache_path is None:
            parent = os.path.dirname(os.path.abspath(transcriptions_dir))
            cache_path = os.path.join(parent, f"{LEXICAL_CACHE_PREFIX}.{self.hostname}.json")
        self.cache_path = cache_path
        self._months: dict[str, dict[str, Any]] = {}
        self._daily: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._loaded_from_disk = False
        self._last_entry_ts: dt.datetime | None = None
        self._last_entry_month: str | None = None
        self._today_token_tail: dict[str, list[str]] = {"en": [], "de": []}
        self._today_tail_date: str | None = None
        self.scanned_files: int = 0
        self.rebuilt_months: int = 0

    @property
    def daily(self) -> dict[str, dict[str, Any]]:
        return self._daily

    def _load_cache_file(self) -> None:
        if self._loaded_from_disk:
            return
        self._loaded_from_disk = True
        if not os.path.isfile(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and payload.get("schema_version") == CACHE_SCHEMA_VERSION:
                months = payload.get("months")
                if isinstance(months, dict):
                    self._months = {
                        k: v for k, v in months.items()
                        if isinstance(k, str) and isinstance(v, dict)
                    }
        except Exception:
            self._months = {}

    def save(self) -> None:
        """Atomically persist the cache if modified (with Windows Dropbox/AV lock retry)."""
        if not self._dirty:
            return
        parent = os.path.dirname(os.path.abspath(self.cache_path))
        os.makedirs(parent, exist_ok=True)
        tmp_path = f"{self.cache_path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "months": self._months,
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            for attempt in range(5):
                try:
                    os.replace(tmp_path, self.cache_path)
                    self._dirty = False
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    import time as _time
                    _time.sleep(0.02 * (attempt + 1))
        except OSError:
            pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def refresh(self, force_rebuild: bool = False, persist: bool = True) -> dict[str, dict[str, Any]]:
        """Incrementally sync against ``transcriptions/`` and ``transcriptions/meta/``."""
        if force_rebuild:
            self._months = {}
            self._loaded_from_disk = True
        else:
            self._load_cache_file()

        month_txt_files: dict[str, list[tuple[str, str, int, int]]] = collections.defaultdict(list)
        month_meta_files: dict[str, list[tuple[str, str, int, int]]] = collections.defaultdict(list)
        scanned = 0

        if os.path.isdir(self.transcriptions_dir):
            try:
                with os.scandir(self.transcriptions_dir) as it:
                    for dent in it:
                        name = dent.name
                        low = name.lower()
                        if not low.endswith(".txt") or "conflicted copy" in low:
                            continue
                        try:
                            st_info = dent.stat()
                        except OSError:
                            continue
                        if not os.path.isfile(dent.path):
                            continue
                        scanned += 1
                        mkey = _month_bucket_key(name)
                        month_txt_files[mkey].append(
                            (name, dent.path, int(st_info.st_size), int(st_info.st_mtime_ns))
                        )
            except OSError:
                pass

            meta_dir = os.path.join(self.transcriptions_dir, ACOUSTIC_META_SUBDIR)
            if os.path.isdir(meta_dir):
                try:
                    with os.scandir(meta_dir) as it:
                        for dent in it:
                            name = dent.name
                            low = name.lower()
                            if not low.endswith(".jsonl") or "conflicted copy" in low:
                                continue
                            try:
                                st_info = dent.stat()
                            except OSError:
                                continue
                            mkey = _month_bucket_key(name)
                            month_meta_files[mkey].append(
                                (f"meta/{name}", dent.path, int(st_info.st_size), int(st_info.st_mtime_ns))
                            )
                except OSError:
                    pass

        self.scanned_files = scanned
        all_month_keys = sorted(set(month_txt_files.keys()) | set(month_meta_files.keys()))

        # Drop cached months whose source files were deleted
        removed = set(self._months.keys()) - set(all_month_keys)
        if removed:
            for k in removed:
                self._months.pop(k, None)
            self._dirty = True

        rebuilt = 0
        for mkey in all_month_keys:
            txt_list = sorted(month_txt_files.get(mkey, []), key=lambda x: x[0])
            meta_list = sorted(month_meta_files.get(mkey, []), key=lambda x: x[0])
            expected_sig = {
                rel: [sz, mt] for rel, _p, sz, mt in (txt_list + meta_list)
            }
            cached_month = self._months.get(mkey)
            if (
                not force_rebuild
                and isinstance(cached_month, dict)
                and cached_month.get("files") == expected_sig
                and isinstance(cached_month.get("days"), dict)
            ):
                continue

            # Parse only this changed/new month
            rebuilt += 1
            # Compute burst gaps per-file (each machine's stream is sequential)
            # while merging lexical & acoustic features by calendar day.
            month_entries_by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            file_gaps_by_day: dict[str, list[float]] = collections.defaultdict(list)

            for _rel, path, _sz, _mt in txt_list:
                f_entries = parse_transcript_file(path)
                f_entries.sort(key=lambda e: e["timestamp"])
                prev_ts: dt.datetime | None = None
                for e in f_entries:
                    ts: dt.datetime = e["timestamp"]
                    day_iso = ts.strftime("%Y-%m-%d")
                    month_entries_by_day[day_iso].append(e)
                    if prev_ts is not None:
                        gap = (ts - prev_ts).total_seconds()
                        if 0.0 < gap <= BURST_GAP_MAX_SECONDS:
                            file_gaps_by_day[day_iso].append(gap)
                    prev_ts = ts

            month_ac_by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for _rel, path, _sz, _mt in meta_list:
                for d_iso, recs in parse_acoustic_meta_file(path).items():
                    month_ac_by_day[d_iso].extend(recs)

            month_days: dict[str, dict[str, Any]] = {}
            all_days_in_month = sorted(set(month_entries_by_day.keys()) | set(month_ac_by_day.keys()))
            for day_iso in all_days_in_month:
                d_entries = sorted(month_entries_by_day.get(day_iso, []), key=lambda e: e["timestamp"])
                # Pass prev_ts=None and apply exact per-machine file_gaps_by_day below
                rec, _ = _fold_day_entries_without_gaps(d_entries, month_ac_by_day.get(day_iso))
                for gap in file_gaps_by_day.get(day_iso, []):
                    rec["gap_count"] += 1
                    rec["gap_sum_s"] = round(rec["gap_sum_s"] + gap, 3)
                    if gap < 2.0:
                        rec["gap_short"] += 1
                    elif gap < 5.0:
                        rec["gap_med"] += 1
                    else:
                        rec["gap_long"] += 1
                        if gap >= 10.0:
                            rec["gap_vlong"] += 1
                    rec["gap_hist"][_pause_bin_index(gap)] += 1
                month_days[day_iso] = rec

            self._months[mkey] = {
                "files": expected_sig,
                "days": month_days,
            }
            self._dirty = True

        self.rebuilt_months = rebuilt
        self._rebuild_daily_union()
        if persist and self._dirty:
            self.save()
        return self._daily

    def _rebuild_daily_union(self) -> None:
        combined: dict[str, dict[str, Any]] = {}
        for mkey in sorted(self._months.keys()):
            m_obj = self._months.get(mkey)
            if not isinstance(m_obj, dict):
                continue
            days_map = m_obj.get("days")
            if not isinstance(days_map, dict):
                continue
            for day_iso, rec in days_map.items():
                if not isinstance(rec, dict) or not _is_iso_date(day_iso):
                    continue
                if day_iso not in combined:
                    combined[day_iso] = _empty_day_record()
                _merge_day_records(combined[day_iso], rec)
        self._daily = dict(sorted(combined.items()))

    def _seed_day_tail_from_disk(self, mkey: str, day_iso: str) -> None:
        """Seed ``_today_token_tail`` and per-host ``_last_entry_ts`` from existing month ``.txt`` files."""
        self._today_tail_date = day_iso
        self._today_token_tail = {"en": [], "de": []}
        if self._last_entry_month != mkey:
            self._last_entry_ts = None
            self._last_entry_month = mkey
        if not os.path.isdir(self.transcriptions_dir):
            return
        day_entries: list[dict[str, Any]] = []
        host_entries: list[dict[str, Any]] = []
        expected_host_file = f"{mkey}.{self.hostname}.txt".lower()
        try:
            for name in sorted(os.listdir(self.transcriptions_dir)):
                low = name.lower()
                if not low.endswith(".txt") or "conflicted copy" in low:
                    continue
                if _month_bucket_key(name) != mkey:
                    continue
                fpath = os.path.join(self.transcriptions_dir, name)
                if not os.path.isfile(fpath):
                    continue
                parsed = parse_transcript_file(fpath)
                for entry in parsed:
                    if entry["timestamp"].strftime("%Y-%m-%d") == day_iso:
                        day_entries.append(entry)
                    if low == expected_host_file or low == f"{mkey}.txt":
                        host_entries.append(entry)
        except OSError:
            return

        if host_entries:
            host_entries.sort(key=lambda e: e["timestamp"])
            self._last_entry_ts = host_entries[-1]["timestamp"]
            self._last_entry_month = mkey

        if day_entries:
            day_entries.sort(key=lambda e: e["timestamp"])
            by_lang: dict[str, list[str]] = {"en": [], "de": []}
            for entry in day_entries:
                toks = tokenize_words(entry["text"])
                if toks:
                    lang = detect_entry_language(entry["text"], toks)
                    by_lang[lang].extend(toks)
            for lang in ("en", "de"):
                self._today_token_tail[lang] = by_lang[lang][-MATTR_WINDOW:]

    def record_live_entry(
        self,
        timestamp: dt.datetime,
        text: str,
        acoustic_meta: dict[str, Any] | None = None,
    ) -> None:
        """O(1) in-memory update when a new dictation is saved during a live session.

        Call *before* appending the new line to ``YYYY-MM.HOST.txt`` (then call
        ``sync_host_watermarks(timestamp)`` afterwards) when mirroring disk writes,
        or call standalone for pure in-memory streaming updates.
        """
        if not text or not text.strip():
            return
        if not self._loaded_from_disk and not self._daily:
            self.refresh(persist=False)

        mkey = timestamp.strftime("%Y-%m")
        day_iso = timestamp.strftime("%Y-%m-%d")
        month_slot = self._months.setdefault(mkey, {"files": {}, "days": {}})
        month_days = month_slot.setdefault("days", {})

        if self._today_tail_date != day_iso:
            self._seed_day_tail_from_disk(mkey, day_iso)
        elif self._last_entry_month != mkey:
            self._last_entry_ts = None
            self._last_entry_month = mkey

        toks = tokenize_words(text)
        n = len(toks)
        lang = detect_entry_language(text, toks)

        target_rec = month_days.setdefault(day_iso, _empty_day_record())
        target_rec["entries"] += 1
        target_rec["words"] += n
        target_rec["hours"][timestamp.hour] += n

        if self._last_entry_ts is not None and self._last_entry_month == mkey:
            gap = (timestamp - self._last_entry_ts).total_seconds()
            if 0.0 < gap <= BURST_GAP_MAX_SECONDS:
                target_rec["gap_count"] += 1
                target_rec["gap_sum_s"] = round(target_rec["gap_sum_s"] + gap, 3)
                if gap < 2.0:
                    target_rec["gap_short"] += 1
                elif gap < 5.0:
                    target_rec["gap_med"] += 1
                else:
                    target_rec["gap_long"] += 1
                    if gap >= 10.0:
                        target_rec["gap_vlong"] += 1
                target_rec["gap_hist"][_pause_bin_index(gap)] += 1
        self._last_entry_ts = timestamp
        self._last_entry_month = mkey

        if n > 0:
            feats = _count_entry_features(text, toks, lang)
            lb = target_rec[lang]
            prev_lang_words = int(lb["words"])
            lb["entries"] += 1
            lb["words"] = prev_lang_words + n
            for fk, fv in feats.items():
                lb[fk] = int(lb[fk]) + fv

            # Exact O(1) sliding MATTR-50 update (handles <50 -> >=50 transition cleanly)
            tail = self._today_token_tail[lang]
            if prev_lang_words < MATTR_WINDOW and len(tail) != prev_lang_words:
                tail = tail[-prev_lang_words:] if prev_lang_words > 0 else []
                self._today_token_tail[lang] = tail
            running_count = prev_lang_words
            for tok in toks:
                running_count += 1
                tail.append(tok)
                if len(tail) > MATTR_WINDOW:
                    tail.pop(0)
                if running_count == MATTR_WINDOW and len(tail) == MATTR_WINDOW:
                    lb["mattr_sum"] = round(len(set(tail)) / float(MATTR_WINDOW), 4)
                    lb["mattr_w"] = 1.0
                elif running_count > MATTR_WINDOW and len(tail) == MATTR_WINDOW:
                    lb["mattr_sum"] = round(float(lb["mattr_sum"]) + (len(set(tail)) / float(MATTR_WINDOW)), 4)
                    lb["mattr_w"] = round(float(lb["mattr_w"]) + 1.0, 4)

            if int(lb["words"]) < MATTR_WINDOW and len(tail) >= 8:
                m_sum, m_w = compute_mattr(tail, MATTR_WINDOW)
                lb["mattr_sum"] = round(m_sum, 4)
                lb["mattr_w"] = round(m_w, 4)

        if acoustic_meta:
            ac_copy = dict(acoustic_meta)
            if "words" not in ac_copy or ac_copy["words"] is None:
                ac_copy["words"] = n
            _accumulate_acoustic_entry(target_rec, ac_copy)

        # Recompute this day's union across _months in O(1) and mark cache dirty
        merged_day = _empty_day_record()
        for m_obj in self._months.values():
            if not isinstance(m_obj, dict):
                continue
            d_rec = (m_obj.get("days") or {}).get(day_iso)
            if isinstance(d_rec, dict):
                _merge_day_records(merged_day, d_rec)
        was_new_day = day_iso not in self._daily
        self._daily[day_iso] = merged_day
        if was_new_day:
            self._daily = dict(sorted(self._daily.items()))
        self._dirty = True

    def sync_host_watermarks(self, timestamp: dt.datetime) -> None:
        """Update cached ``(size, mtime_ns)`` watermarks for this machine's current month files."""
        mkey = timestamp.strftime("%Y-%m")
        month_slot = self._months.get(mkey)
        if not isinstance(month_slot, dict):
            return
        files_sig = month_slot.setdefault("files", {})
        txt_name = f"{mkey}.{self.hostname}.txt"
        txt_path = os.path.join(self.transcriptions_dir, txt_name)
        if os.path.isfile(txt_path):
            try:
                st_info = os.stat(txt_path)
                files_sig[txt_name] = [int(st_info.st_size), int(st_info.st_mtime_ns)]
                self._dirty = True
            except OSError:
                pass
        meta_name = f"{mkey}.{self.hostname}.jsonl"
        meta_path = os.path.join(self.transcriptions_dir, ACOUSTIC_META_SUBDIR, meta_name)
        if os.path.isfile(meta_path):
            try:
                st_info = os.stat(meta_path)
                files_sig[f"meta/{meta_name}"] = [int(st_info.st_size), int(st_info.st_mtime_ns)]
                self._dirty = True
            except OSError:
                pass


def _fold_day_entries_without_gaps(
    day_entries: list[dict[str, Any]],
    day_acoustic: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], None]:
    """Fold lexical & acoustic features for a day without cross-file gap mixing."""
    rec = _empty_day_record()
    lang_tokens: dict[str, list[str]] = {"en": [], "de": []}

    for entry in day_entries:
        ts: dt.datetime = entry["timestamp"]
        text: str = entry["text"]
        toks = tokenize_words(text)
        n = len(toks)
        lang = detect_entry_language(text, toks)

        rec["entries"] += 1
        rec["words"] += n
        rec["hours"][ts.hour] += n
        if n == 0:
            continue

        lang_tokens[lang].extend(toks)
        feats = _count_entry_features(text, toks, lang)
        lb = rec[lang]
        lb["entries"] += 1
        lb["words"] += n
        for fk, fv in feats.items():
            lb[fk] += fv

    for lang in ("en", "de"):
        m_sum, m_w = compute_mattr(lang_tokens[lang], MATTR_WINDOW)
        rec[lang]["mattr_sum"] = round(m_sum, 4)
        rec[lang]["mattr_w"] = round(m_w, 4)

    if day_acoustic:
        for ac in day_acoustic:
            _accumulate_acoustic_entry(rec, ac)

    return rec, None


# ---------------------------------------------------------------------------
# Multi-Horizon Date Slicing, Language Alignment & Metric Computation
# ---------------------------------------------------------------------------

def resolve_date_bounds(
    range_spec: Any,
    today: dt.date | None = None,
    first_archive_date: dt.date | None = None,
) -> tuple[dt.date | None, dt.date | None]:
    """Resolve a human label or spec into inclusive ``(start_date, end_date)``."""
    today = today or dt.date.today()

    spec = INSIGHTS_RANGE_OPTIONS.get(range_spec, range_spec) if isinstance(range_spec, str) else range_spec

    if spec is None or spec == "All" or spec == "All Time":
        if first_archive_date is not None and first_archive_date > today:
            return today, first_archive_date
        return first_archive_date, today
    if spec == 0 or spec == "Today":
        return today, today
    if isinstance(spec, int) and spec > 0:
        return today - dt.timedelta(days=spec - 1), today
    if isinstance(spec, str):
        s = spec.strip()
        low = s.lower()
        if low in ("ytd", "this year"):
            return dt.date(today.year, 1, 1), today
        if low in ("prev_6m", "prev 6 months", "previous 6 months"):
            end_d = today - dt.timedelta(days=182)
            start_d = today - dt.timedelta(days=364)
            return start_d, end_d
        if low in ("last_quarter", "last quarter"):
            return today - dt.timedelta(days=90), today
        if low == "previous month":
            last_prev = today.replace(day=1) - dt.timedelta(days=1)
            return last_prev.replace(day=1), last_prev
        if low == "last 7 days":
            return today - dt.timedelta(days=6), today
        if low == "last 30 days":
            return today - dt.timedelta(days=29), today
        if low.startswith("year:") or low.startswith("year "):
            yr_str = s.split(":", 1)[1] if ":" in s else s.split(None, 1)[1]
            try:
                yr = int(yr_str.strip())
                start_d = dt.date(yr, 1, 1)
                end_d = min(today, dt.date(yr, 12, 31)) if yr <= today.year else dt.date(yr, 12, 31)
                return start_d, end_d
            except ValueError:
                pass
        if len(s) == 4 and s.isascii() and s.isdigit():
            try:
                yr = int(s)
                start_d = dt.date(yr, 1, 1)
                end_d = min(today, dt.date(yr, 12, 31)) if yr <= today.year else dt.date(yr, 12, 31)
                return start_d, end_d
            except ValueError:
                pass
        if low.startswith("month:"):
            ym = s.split(":", 1)[1].strip()
            try:
                yr, mo = (int(p) for p in ym.split("-", 1))
                start_d = dt.date(yr, mo, 1)
                next_m = dt.date(yr + (1 if mo == 12 else 0), 1 if mo == 12 else mo + 1, 1)
                last_m = next_m - dt.timedelta(days=1)
                end_d = min(today, last_m) if start_d <= today else last_m
                return start_d, end_d
            except ValueError:
                pass
        if ".." in s:
            raw_s = s.split(":", 1)[1] if low.startswith("range:") else s
            p0, p1 = raw_s.split("..", 1)
            try:
                d0 = dt.date.fromisoformat(p0.strip())
                d1 = dt.date.fromisoformat(p1.strip())
                return (d0, d1) if d0 <= d1 else (d1, d0)
            except ValueError:
                pass

    return today - dt.timedelta(days=181), today


def available_period_options(daily_map: dict[str, Any], today: dt.date | None = None) -> list[str]:
    """Return all selectable period labels including archive calendar years."""
    today = today or dt.date.today()
    base = [
        "Today",
        "7 days",
        "30 days",
        "90 days",
        "6 months",
        "This year",
        "12 months",
        "Prev 6 months",
        "All",
    ]
    years = sorted({d[:4] for d in daily_map.keys() if len(d) >= 4 and d[:4].isdigit()}, reverse=True)
    for yr in years:
        label = f"Year {yr}"
        if label not in base:
            base.append(label)
    return base


def _compute_language_offsets(daily_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute speaker-specific EN vs DE mean offsets so mixed-language 'All'
    trajectories never jump artificially when switching between German & English."""
    sums = {
        "en": {"mattr_num": 0.0, "mattr_den": 0.0, "content_num": 0, "long_num": 0, "words": 0},
        "de": {"mattr_num": 0.0, "mattr_den": 0.0, "content_num": 0, "long_num": 0, "words": 0},
    }
    for rec in daily_map.values():
        for lang in ("en", "de"):
            lb = rec.get(lang) or {}
            w = int(lb.get("words", 0))
            if w <= 0:
                continue
            sums[lang]["words"] += w
            sums[lang]["content_num"] += max(0, w - int(lb.get("stop_words", 0)))
            sums[lang]["long_num"] += int(lb.get("long_words", 0))
            sums[lang]["mattr_num"] += float(lb.get("mattr_sum", 0.0))
            sums[lang]["mattr_den"] += float(lb.get("mattr_w", 0.0))

    total_w = sums["en"]["words"] + sums["de"]["words"]
    if sums["en"]["words"] < 100 or sums["de"]["words"] < 100 or total_w == 0:
        return {"en": {"mattr": 0.0, "content": 0.0, "long": 0.0},
                "de": {"mattr": 0.0, "content": 0.0, "long": 0.0}}

    means: dict[str, dict[str, float]] = {}
    has_both_mattr = sums["en"]["mattr_den"] > 0 and sums["de"]["mattr_den"] > 0
    for lang in ("en", "de"):
        s = sums[lang]
        mattr_m = (s["mattr_num"] / s["mattr_den"] * 100.0) if s["mattr_den"] > 0 else 0.0
        content_m = (s["content_num"] / s["words"] * 100.0) if s["words"] > 0 else 54.0
        long_m = (s["long_num"] / s["words"] * 100.0) if s["words"] > 0 else 20.0
        means[lang] = {"mattr": mattr_m, "content": content_m, "long": long_m}

    w_en = sums["en"]["words"] / float(total_w)
    w_de = sums["de"]["words"] / float(total_w)
    ref = {
        k: w_en * means["en"][k] + w_de * means["de"][k]
        for k in ("mattr", "content", "long")
    }
    return {
        lang: {
            "mattr": (ref["mattr"] - means[lang]["mattr"]) if has_both_mattr else 0.0,
            "content": ref["content"] - means[lang]["content"],
            "long": ref["long"] - means[lang]["long"],
        }
        for lang in ("en", "de")
    }


def compute_record_metrics(
    rec: dict[str, Any],
    lang_mode: str = "All",
    lang_offsets: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Convert a raw sufficient-statistics record (day/week/month/window) into metrics."""
    mode = (lang_mode or "All").strip().lower()
    langs = ("en",) if mode == "en" else (("de",) if mode == "de" else ("en", "de"))
    offsets = lang_offsets or {
        "en": {"mattr": 0.0, "content": 0.0, "long": 0.0},
        "de": {"mattr": 0.0, "content": 0.0, "long": 0.0},
    }

    words = sum(int((rec.get(l) or {}).get("words", 0)) for l in langs)
    entries = sum(int((rec.get(l) or {}).get("entries", 0)) for l in langs)
    if mode == "all" and entries == 0:
        entries = int(rec.get("entries", 0))

    if mode == "en":
        de_share = 0.0
    elif mode == "de":
        de_share = 100.0 if words > 0 else 0.0
    else:
        tot_en_de = int((rec.get("en") or {}).get("words", 0)) + int((rec.get("de") or {}).get("words", 0))
        de_words = int((rec.get("de") or {}).get("words", 0))
        de_share = (de_words / float(tot_en_de)) * 100.0 if tot_en_de > 0 else 0.0

    sentences = sum(int((rec.get(l) or {}).get("sentences", 0)) for l in langs)
    fillers = sum(int((rec.get(l) or {}).get("fillers", 0)) for l in langs)
    repeats = sum(int((rec.get(l) or {}).get("repeats", 0)) for l in langs)
    aborts = sum(int((rec.get(l) or {}).get("aborts", 0)) for l in langs)
    vague = sum(int((rec.get(l) or {}).get("vague", 0)) for l in langs)
    pronoun_i = sum(int((rec.get(l) or {}).get("pronoun_i", 0)) for l in langs)
    pronoun_we = sum(int((rec.get(l) or {}).get("pronoun_we", 0)) for l in langs)
    absolutist = sum(int((rec.get(l) or {}).get("absolutist", 0)) for l in langs)
    hedges = sum(int((rec.get(l) or {}).get("hedges", 0)) for l in langs)
    pos_vit = sum(int((rec.get(l) or {}).get("pos_vitality", 0)) for l in langs)
    neg_str = sum(int((rec.get(l) or {}).get("neg_strain", 0)) for l in langs)
    codeswitch = sum(int((rec.get(l) or {}).get("codeswitch_hits", 0)) for l in langs)

    # MATTR-50, content-word density & long-word share (language-aligned when mode == 'all')
    mattr_val: float | None = None
    content_pct: float | None = None
    long_pct: float | None = None

    if words > 0:
        if mode == "all":
            m_weighted = 0.0
            m_w_tot = 0.0
            c_weighted = 0.0
            l_weighted = 0.0
            for l in ("en", "de"):
                lb = rec.get(l) or {}
                lw = int(lb.get("words", 0))
                if lw <= 0:
                    continue
                ms = float(lb.get("mattr_sum", 0.0))
                mw = float(lb.get("mattr_w", 0.0))
                if mw > 0.0:
                    l_mattr = (ms / mw) * 100.0 + offsets[l]["mattr"]
                    m_weighted += l_mattr * mw
                    m_w_tot += mw
                l_cont = ((lw - int(lb.get("stop_words", 0))) / float(lw)) * 100.0 + offsets[l]["content"]
                l_long = (int(lb.get("long_words", 0)) / float(lw)) * 100.0 + offsets[l]["long"]
                c_weighted += l_cont * lw
                l_weighted += l_long * lw
            if m_w_tot > 0.0:
                mattr_val = max(5.0, min(98.0, m_weighted / m_w_tot))
            content_pct = max(5.0, min(95.0, c_weighted / float(words)))
            long_pct = max(0.0, min(80.0, l_weighted / float(words)))
        else:
            lb = rec.get(langs[0]) or {}
            ms = float(lb.get("mattr_sum", 0.0))
            mw = float(lb.get("mattr_w", 0.0))
            if mw > 0.0:
                mattr_val = max(5.0, min(98.0, (ms / mw) * 100.0))
            content_pct = max(5.0, min(95.0, ((words - int(lb.get("stop_words", 0))) / float(words)) * 100.0))
            long_pct = max(0.0, min(80.0, (int(lb.get("long_words", 0)) / float(words)) * 100.0))

    # Pause metrics (burst-gap timestamps)
    gap_count = int(rec.get("gap_count", 0))
    pause_avg_s = (float(rec.get("gap_sum_s", 0.0)) / gap_count) if gap_count > 0 else None
    pause_long_pct = (int(rec.get("gap_long", 0)) / float(gap_count) * 100.0) if gap_count > 0 else None
    pause_p90_s = (
        quantile_from_pause_hist(rec.get("gap_hist") or [], gap_count, 0.90)
        if gap_count > 0
        else None
    )

    # Circadian late-night share (22:00 - 04:59)
    hours = rec.get("hours") or [0] * 24
    tot_hour_words = sum(hours)
    late_words = sum(hours[h] for h in (22, 23, 0, 1, 2, 3, 4)) if len(hours) == 24 else 0
    late_night_pct = (late_words / float(tot_hour_words) * 100.0) if tot_hour_words > 0 else None

    # Forward acoustic timing metrics (when present)
    ac_audio_s = float(rec.get("ac_audio_s", 0.0) or 0.0)
    ac_speech_s = float(rec.get("ac_speech_s", 0.0) or 0.0)
    ac_words = int(rec.get("ac_words", 0) or 0)
    acoustic_wpm = (ac_words / (ac_speech_s / 60.0)) if (ac_speech_s >= 2.0 and ac_words > 0) else None
    acoustic_silence_pct = (
        max(0.0, (ac_audio_s - ac_speech_s) / ac_audio_s * 100.0)
        if ac_audio_s >= 2.0
        else None
    )

    if words > 0:
        fillers_per_100 = (fillers * 100.0) / words
        disfluency_per_1k = ((repeats + aborts) * 1000.0) / words
        repetition_per_1k = (repeats * 1000.0) / words
        vague_per_100 = (vague * 100.0) / words
        self_focus_per_100 = (pronoun_i * 100.0) / words
        we_focus_per_100 = (pronoun_we * 100.0) / words
        absolutist_per_100 = (absolutist * 100.0) / words
        hedges_per_100 = (hedges * 100.0) / words
        pos_rate = (pos_vit * 100.0) / words
        neg_rate = (neg_str * 100.0) / words
        vitality_balance = max(5.0, min(95.0, 50.0 + 12.0 * (pos_rate - neg_rate)))
        codeswitch_pct = (codeswitch * 100.0) / words
        pace_words_entry = words / float(max(1, entries))
        words_per_sentence = words / float(max(1, sentences))
    else:
        fillers_per_100 = None
        disfluency_per_1k = None
        repetition_per_1k = None
        vague_per_100 = None
        self_focus_per_100 = None
        we_focus_per_100 = None
        absolutist_per_100 = None
        hedges_per_100 = None
        vitality_balance = None
        codeswitch_pct = None
        pace_words_entry = None
        words_per_sentence = None

    return {
        "words": words,
        "entries": entries,
        "de_share_pct": round(de_share, 1),
        "low_confidence": words < INSIGHTS_MIN_WORDS_CONFIDENT,
        "volume_words": float(words),
        "mattr_diversity": round(mattr_val, 2) if mattr_val is not None else None,
        "content_density_pct": round(content_pct, 2) if content_pct is not None else None,
        "lexical_long_word_percent": round(long_pct, 2) if long_pct is not None else None,
        "vague_per_100": round(vague_per_100, 3) if vague_per_100 is not None else None,
        "fillers_per_100": round(fillers_per_100, 3) if fillers_per_100 is not None else None,
        "disfluency_per_1k": round(disfluency_per_1k, 2) if disfluency_per_1k is not None else None,
        "repetition_per_1k": round(repetition_per_1k, 2) if repetition_per_1k is not None else None,
        "pause_avg_s": round(pause_avg_s, 2) if pause_avg_s is not None else None,
        "pause_long_pct": round(pause_long_pct, 2) if pause_long_pct is not None else None,
        "pause_p90_s": round(pause_p90_s, 2) if pause_p90_s is not None else None,
        "pace_words_entry": round(pace_words_entry, 2) if pace_words_entry is not None else None,
        "words_per_sentence": round(words_per_sentence, 2) if words_per_sentence is not None else None,
        "self_focus_per_100": round(self_focus_per_100, 3) if self_focus_per_100 is not None else None,
        "we_focus_per_100": round(we_focus_per_100, 3) if we_focus_per_100 is not None else None,
        "absolutist_per_100": round(absolutist_per_100, 3) if absolutist_per_100 is not None else None,
        "hedges_per_100": round(hedges_per_100, 3) if hedges_per_100 is not None else None,
        "vitality_balance": round(vitality_balance, 2) if vitality_balance is not None else None,
        "late_night_pct": round(late_night_pct, 2) if late_night_pct is not None else None,
        "codeswitch_pct": round(codeswitch_pct, 2) if codeswitch_pct is not None else None,
        "acoustic_wpm": round(acoustic_wpm, 1) if acoustic_wpm is not None else None,
        "acoustic_silence_pct": round(acoustic_silence_pct, 2) if acoustic_silence_pct is not None else None,
        # Raw pause breakdown counts for detail display
        "gap_short": int(rec.get("gap_short", 0)),
        "gap_med": int(rec.get("gap_med", 0)),
        "gap_long": int(rec.get("gap_long", 0)),
        "gap_count": gap_count,
    }


def _robust_center_and_scale(values: list[float], fallback_low: float, fallback_high: float) -> tuple[float, float, float, float, bool]:
    """Return ``(median, sigma_mad, band_low, band_high, is_personal)``."""
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    min_sigma = max(0.15, abs(fallback_high - fallback_low) * 0.08)
    if len(clean) < PERSONAL_BASELINE_MIN_DAYS:
        mid = (fallback_low + fallback_high) / 2.0
        sigma = max(min_sigma, abs(fallback_high - fallback_low) / 2.5)
        return mid, sigma, fallback_low, fallback_high, False

    med = float(statistics.median(clean))
    abs_devs = [abs(v - med) for v in clean]
    mad = float(statistics.median(abs_devs)) * 1.4826
    if mad < min_sigma:
        sorted_v = sorted(clean)
        q1 = sorted_v[len(sorted_v) // 4]
        q3 = sorted_v[(len(sorted_v) * 3) // 4]
        mad = max(mad, (q3 - q1) / 1.349, abs(med) * 0.08, min_sigma)
    low = med - 1.15 * mad
    high = med + 1.15 * mad
    return med, mad, low, high, True


def compute_personal_baselines(
    daily_map: dict[str, dict[str, Any]],
    lang_mode: str = "All",
    lang_offsets: dict[str, dict[str, float]] | None = None,
    today: dt.date | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Compute trailing 90-day robust personal baseline bands per metric."""
    today = today or dt.date.today()
    cutoff_iso = (today - dt.timedelta(days=PERSONAL_BASELINE_DAYS)).isoformat()
    today_iso = today.isoformat()

    confident_days: list[dict[str, Any]] = []
    for day_iso, rec in daily_map.items():
        if cutoff_iso <= day_iso <= today_iso and isinstance(rec, dict):
            m = compute_record_metrics(rec, lang_mode=lang_mode, lang_offsets=lang_offsets)
            if m["words"] >= INSIGHTS_MIN_WORDS_CONFIDENT:
                confident_days.append(m)

    # If user was inactive in last 90d but has rich older archive, fall back to last 60 confident days <= today
    if len(confident_days) < PERSONAL_BASELINE_MIN_DAYS and len(daily_map) >= PERSONAL_BASELINE_MIN_DAYS:
        all_conf = []
        for _day_iso, rec in sorted(daily_map.items()):
            if _day_iso > today_iso or not isinstance(rec, dict):
                continue
            m = compute_record_metrics(rec, lang_mode=lang_mode, lang_offsets=lang_offsets)
            if m["words"] >= INSIGHTS_MIN_WORDS_CONFIDENT:
                all_conf.append(m)
        if len(all_conf) >= PERSONAL_BASELINE_MIN_DAYS:
            confident_days = all_conf[-60:]

    baselines: dict[str, dict[str, Any]] = {}
    for mkey, spec in METRIC_SPECS.items():
        if mkey == "clarity_index":
            continue
        peer_key = spec.get("peer_key")
        peer = PEER_BENCHMARKS.get(peer_key or "", {"low": 0.0, "high": 100.0})
        f_low = float(peer.get("low", 0.0))
        f_high = float(peer.get("high", 100.0))
        vals = [d[mkey] for d in confident_days if d.get(mkey) is not None]
        med, sigma, b_low, b_high, is_personal = _robust_center_and_scale(vals, f_low, f_high)
        if f_low >= 0.0:
            b_low = max(0.0, b_low)
        if spec.get("unit") == "%":
            b_high = min(100.0, b_high)
        baselines[mkey] = {
            "median": round(med, 3),
            "sigma": round(sigma, 3),
            "low": round(b_low, 3),
            "high": round(b_high, 3),
            "is_personal": is_personal,
            "n_days": len(vals),
        }

    baselines["clarity_index"] = {
        "median": 50.0,
        "sigma": 10.0,
        "low": 40.0,
        "high": 60.0,
        "is_personal": len(confident_days) >= PERSONAL_BASELINE_MIN_DAYS,
        "n_days": len(confident_days),
    }
    return baselines, len(confident_days)


def score_clarity_index(metrics: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> float | None:
    """Compute the 0-100 Personal Cognitive Clarity Index (50 = personal norm)."""
    if int(metrics.get("words", 0)) < 8:
        return None

    components: list[tuple[float, float]] = []
    spec_weights = (
        ("mattr_diversity", +1.0, 1.25),
        ("vague_per_100", -1.0, 1.0),
        ("fillers_per_100", -1.0, 0.85),
        ("disfluency_per_1k", -1.0, 0.85),
        ("pause_long_pct", -1.0, 1.0),
        ("pace_words_entry", +1.0, 0.5),
    )
    for mkey, sign, weight in spec_weights:
        val = metrics.get(mkey)
        base = baselines.get(mkey)
        if val is None or not base:
            continue
        f_val = float(val)
        f_med = float(base.get("median", 0.0))
        sigma = max(float(base.get("sigma", 1.0)), 1e-3)
        if not (math.isfinite(f_val) and math.isfinite(f_med) and math.isfinite(sigma)):
            continue
        z = sign * ((f_val - f_med) / sigma)
        z = max(-3.0, min(3.0, z))
        components.append((z, weight))

    if not components:
        return None
    w_sum = sum(w for _, w in components)
    mean_z = sum(z * w for z, w in components) / max(w_sum, 1e-6)
    return round(max(5.0, min(95.0, 50.0 + 10.0 * mean_z)), 1)


def _choose_bucket_mode(bucket_mode: str, start_d: dt.date | None, end_d: dt.date | None) -> str:
    mode = (bucket_mode or "Auto").strip().lower()
    if mode in ("daily", "weekly", "monthly", "hourly"):
        return mode
    if start_d is not None and end_d is not None:
        span = max(0, (end_d - start_d).days)
        if span <= 170:
            return "daily"
        if span <= 550:
            return "weekly"
    return "monthly"


def _bucket_key_and_label(day_iso: str, bucket_mode: str) -> tuple[str, str]:
    if bucket_mode == "monthly":
        return day_iso[:7], day_iso[:7]
    if bucket_mode == "weekly":
        d = dt.date.fromisoformat(day_iso)
        monday = d - dt.timedelta(days=d.weekday())
        _iso_y, iso_w, _ = d.isocalendar()
        return monday.isoformat(), f"{monday.isoformat()} (W{iso_w:02d})"
    return day_iso, day_iso


def compute_rolling_series(
    points: list[dict[str, Any]],
    metric_key: str,
    window: int,
) -> list[float | None]:
    """Volume-weighted causal-centered rolling smoother for longitudinal clarity."""
    n = len(points)
    if n == 0:
        return []
    if window <= 1:
        return [p.get(metric_key) for p in points]

    half_back = max(1, window - 1)
    out: list[float | None] = []
    for i in range(n):
        lo = max(0, i - half_back)
        num = 0.0
        den = 0.0
        for j in range(lo, i + 1):
            val = points[j].get(metric_key)
            if val is None or not math.isfinite(float(val)):
                continue
            words_j = max(10.0, min(600.0, float(points[j].get("words", 50))))
            decay = 0.85 ** (i - j)
            w = words_j * decay
            num += float(val) * w
            den += w
        out.append(round(num / den, 2) if den > 0.0 else None)
    return out


def _is_iso_date(day_str: str) -> bool:
    if len(day_str) != 10:
        return False
    try:
        dt.date.fromisoformat(day_str)
        return True
    except ValueError:
        return False


def compute_outlier_cutoff(
    values: list[float | None],
    baseline: dict[str, Any] | None = None,
    metric_key: str = "",
    sigma_mult: float = INSIGHTS_OUTLIER_SIGMA,
) -> tuple[float | None, float | None, float, float]:
    """Compute resistant ``(low_cutoff, high_cutoff, center, sd)`` for ``> 2.5σ`` outlier scrubbing.

    Uses a trimmed core standard deviation + MAD/IQR scale so extreme open-mic
    spikes never inflate ``sd`` and mask themselves, while safeguarding ``0``-heavy
    metrics with a scale-appropriate floor.
    """
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    has_personal = bool(baseline and baseline.get("is_personal"))
    if len(clean) < 4 and not has_personal:
        return None, None, 0.0, 1.0

    spec = METRIC_SPECS.get(metric_key, {})
    peer_key = spec.get("peer_key")
    peer = PEER_BENCHMARKS.get(peer_key or "", {})
    if "low" in peer and "high" in peer:
        min_sd = max(0.20, abs(float(peer["high"]) - float(peer["low"])) * 0.12)
    else:
        med_guess = statistics.median(clean) if clean else 50.0
        min_sd = max(1.0, abs(med_guess) * 0.12)

    if len(clean) >= 5:
        s_vals = sorted(clean)
        n = len(s_vals)
        n_hi = max(1, int(n * 0.15))
        n_lo = int(n * 0.10) if n >= 8 else 0
        core = s_vals[n_lo : n - n_hi]
        if len(core) < 3:
            core = s_vals[:-1]
        med = float(statistics.median(s_vals))
        mad_sd = float(statistics.median([abs(x - med) for x in s_vals])) * 1.4826
        q1 = s_vals[n // 4]
        q3 = s_vals[(n * 3) // 4]
        iqr_sd = (q3 - q1) / 1.349
        core_std = float(statistics.stdev(core)) if len(core) >= 2 else 0.0
        center = (0.5 * med + 0.5 * float(baseline["median"])) if has_personal else med
        sd = max(mad_sd, iqr_sd, core_std, min_sd)
        if has_personal:
            sd = max(sd, float(baseline.get("sigma", min_sd)) * 0.70)
    else:
        center = float(baseline["median"]) if baseline else float(statistics.median(clean))
        sd = max(float(baseline.get("sigma", min_sd)) if baseline else min_sd, min_sd)

    eff_mult = max(1.5, float(sigma_mult))
    high_cut = center + eff_mult * sd
    low_cut: float | None = None
    if spec.get("two_sided_outlier"):
        low_cut = center - eff_mult * sd
    return low_cut, high_cut, center, sd


def filter_outlier_entries(
    entries: list[dict[str, Any]],
    tokens_per_entry: list[list[str]] | None = None,
    sigma_mult: float = INSIGHTS_OUTLIER_SIGMA,
    *,
    outlier_sigma: float | None = None,
) -> tuple[list[int], int]:
    """Identify non-outlier entry indices, excluding accidental open-mic utterances (>2.5σ).

    Catches both runaway long recordings (word count > center + 2.5σ and >= 80 words)
    and silence-hallucination repetition storms (immediate repeat rate > center + 2.5σ).
    Returns ``(kept_indices, removed_count)``.
    """
    eff_sigma = float(outlier_sigma) if outlier_sigma is not None else float(sigma_mult)
    n = len(entries)
    if n < 5:
        return list(range(n)), 0

    toks_list = (
        tokens_per_entry
        if (tokens_per_entry is not None and len(tokens_per_entry) == n)
        else [tokenize_words(e.get("text", "")) for e in entries]
    )
    lengths = [float(len(t)) for t in toks_list]
    non_empty = [x for x in lengths if x > 0]
    if len(non_empty) < 5:
        return list(range(n)), 0

    _, len_high, len_center, _ = compute_outlier_cutoff(
        non_empty, baseline=None, metric_key="pace_words_entry", sigma_mult=eff_sigma
    )
    rep_rates: list[float] = []
    for toks in toks_list:
        if len(toks) >= 6:
            reps = sum(1 for i in range(len(toks) - 1) if toks[i] == toks[i + 1])
            rep_rates.append((reps * 1000.0) / len(toks))
        else:
            rep_rates.append(0.0)
    _, rep_high, _, _ = compute_outlier_cutoff(
        rep_rates, baseline=None, metric_key="disfluency_per_1k", sigma_mult=eff_sigma
    )

    kept: list[int] = []
    removed = 0
    min_runaway_words = max(80.0, len_center * 2.2)
    for idx in range(n):
        w_len = lengths[idx]
        r_rate = rep_rates[idx]
        is_runaway_len = len_high is not None and w_len > len_high and w_len >= min_runaway_words
        is_rep_storm = rep_high is not None and w_len >= 8 and r_rate > rep_high and r_rate >= 180.0
        if is_runaway_len or is_rep_storm:
            removed += 1
        else:
            kept.append(idx)
    return kept, removed


def _detect_runaway_open_mic_days(
    day_metrics_map: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    sigma_mult: float,
) -> set[str]:
    """Detect calendar days dominated by accidental open-mic captures before bucketing."""
    if len(day_metrics_map) < 5:
        return set()

    pace_vals = [m.get("pace_words_entry") for m in day_metrics_map.values()]
    rep_vals = [m.get("disfluency_per_1k") for m in day_metrics_map.values()]
    sil_vals = [m.get("acoustic_silence_pct") for m in day_metrics_map.values()]

    _, pace_hi, pace_c, _ = compute_outlier_cutoff(
        pace_vals, baselines.get("pace_words_entry"), "pace_words_entry", sigma_mult
    )
    _, rep_hi, _, _ = compute_outlier_cutoff(
        rep_vals, baselines.get("disfluency_per_1k"), "disfluency_per_1k", sigma_mult
    )
    _, sil_hi, _, _ = compute_outlier_cutoff(
        sil_vals, baselines.get("acoustic_silence_pct"), "acoustic_silence_pct", sigma_mult
    )

    flagged: set[str] = set()
    for d_iso, m in day_metrics_map.items():
        pw = m.get("pace_words_entry")
        rp = m.get("disfluency_per_1k")
        sl = m.get("acoustic_silence_pct")
        if pw is not None and pace_hi is not None and pw > pace_hi and pw >= max(110.0, pace_c * 2.4):
            flagged.add(d_iso)
        elif rp is not None and rep_hi is not None and rp > rep_hi and rp >= 45.0:
            flagged.add(d_iso)
        elif sl is not None and sil_hi is not None and sl > sil_hi and sl >= 75.0:
            flagged.add(d_iso)
    return flagged


def query_slice(
    index: LexicalDiskIndex | dict[str, dict[str, Any]],
    range_spec: Any = "6 months",
    bucket_mode: str = "Auto",
    lang_mode: str = "All",
    today: dt.date | None = None,
    trim_outliers: bool = False,
    outlier_sigma: float = INSIGHTS_OUTLIER_SIGMA,
) -> dict[str, Any]:
    """Slice, aggregate, optionally scrub >2.5σ outliers, smooth, and summarize in milliseconds."""
    raw_daily = index.daily if isinstance(index, LexicalDiskIndex) else index
    daily_map = {k: v for k, v in (raw_daily or {}).items() if isinstance(k, str) and isinstance(v, dict) and _is_iso_date(k)}
    today = today or dt.date.today()
    eff_sigma = max(1.5, float(outlier_sigma or INSIGHTS_OUTLIER_SIGMA))

    all_dates = sorted(daily_map.keys())
    first_archive = dt.date.fromisoformat(all_dates[0]) if all_dates else None
    start_d, end_d = resolve_date_bounds(range_spec, today=today, first_archive_date=first_archive)

    lang_offsets = _compute_language_offsets(daily_map)
    baselines, baseline_days = compute_personal_baselines(
        daily_map, lang_mode=lang_mode, lang_offsets=lang_offsets, today=today
    )

    resolved_bucket = _choose_bucket_mode(bucket_mode, start_d, end_d)
    start_iso = start_d.isoformat() if start_d else "0000-01-01"
    end_iso = end_d.isoformat() if end_d else "9999-12-31"
    norm_lang = (lang_mode or "All").strip().lower()

    # Optional explicit intra-day hourly curve when bucket_mode == 'hourly' on a single day
    if resolved_bucket == "hourly" and start_d is not None and start_d == end_d:
        day_iso = start_d.isoformat()
        rec = daily_map.get(day_iso) or _empty_day_record()
        summary = compute_record_metrics(rec, lang_mode=lang_mode, lang_offsets=lang_offsets)
        summary["clarity_index"] = score_clarity_index(summary, baselines)
        hours = rec.get("hours") or [0] * 24
        points = []
        for h in range(24):
            hw = int(hours[h]) if h < len(hours) else 0
            pt: dict[str, Any] = {
                "key": f"{h:02d}:00",
                "label": f"{day_iso} {h:02d}:00",
                "short_date": f"{h:02d}h",
                "words": hw,
                "entries": 1 if hw > 0 else 0,
                "de_share_pct": summary["de_share_pct"],
                "low_confidence": hw < INSIGHTS_MIN_WORDS_CONFIDENT,
                "volume_words": float(hw),
            }
            if hw > 0:
                for mkey in METRIC_SPECS:
                    if mkey != "volume_words":
                        pt[mkey] = summary.get(mkey)
            points.append(pt)
        rolling_h = {mkey: compute_rolling_series(points, mkey, 3) for mkey in METRIC_SPECS}
        return {
            "range_spec": range_spec,
            "start_date": start_iso,
            "end_date": end_iso,
            "bucket_mode": "hourly",
            "rolling_window": 3,
            "rolling_label": "3h rolling",
            "lang_mode": lang_mode,
            "trim_outliers": bool(trim_outliers),
            "outlier_sigma": eff_sigma,
            "trimmed_days_count": 0,
            "detected_runaway_days": 0,
            "trimmed_points_by_metric": {},
            "detected_outliers_by_metric": {},
            "total_trimmed_points": 0,
            "total_outlier_buckets": 0,
            "points": points,
            "rolling": rolling_h,
            "summary": summary,
            "baselines": baselines,
            "baseline_days": baseline_days,
            "active_days_in_slice": 1 if summary["words"] > 0 else 0,
            "callout": generate_fogwatch_callout([], summary, baselines, range_spec),
        }

    # Pre-scan matching calendar days
    candidate_days: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for day_iso, rec in daily_map.items():
        if not (start_iso <= day_iso <= end_iso):
            continue
        m_check = compute_record_metrics(rec, lang_mode=lang_mode, lang_offsets=lang_offsets)
        if norm_lang in ("en", "de"):
            if m_check["words"] <= 0:
                continue
        elif m_check["words"] <= 0 and int(rec.get("gap_count", 0)) <= 0:
            continue
        candidate_days[day_iso] = (rec, m_check)

    runaway_days = _detect_runaway_open_mic_days(
        {d: pair[1] for d, pair in candidate_days.items()}, baselines, eff_sigma
    )
    trimmed_days_count = 0

    buckets_raw: dict[str, dict[str, Any]] = {}
    bucket_labels: dict[str, str] = {}
    active_days_in_slice = 0
    window_total_rec = _empty_day_record()

    for day_iso in sorted(candidate_days.keys()):
        rec, _ = candidate_days[day_iso]
        is_trimmed_runaway = (
            trim_outliers
            and day_iso in runaway_days
            and (len(candidate_days) - trimmed_days_count) > 2
        )
        if is_trimmed_runaway:
            trimmed_days_count += 1
            if resolved_bucket != "daily":
                continue
        else:
            active_days_in_slice += 1
            _merge_day_records(window_total_rec, rec)

        bkey, blabel = _bucket_key_and_label(day_iso, resolved_bucket)
        if bkey not in buckets_raw:
            buckets_raw[bkey] = _empty_day_record()
            bucket_labels[bkey] = blabel
        _merge_day_records(buckets_raw[bkey], rec)

    points = []
    for bkey in sorted(buckets_raw.keys()):
        b_metrics = compute_record_metrics(
            buckets_raw[bkey], lang_mode=lang_mode, lang_offsets=lang_offsets
        )
        b_metrics["clarity_index"] = score_clarity_index(b_metrics, baselines)
        b_metrics["key"] = bkey
        b_metrics["label"] = bucket_labels[bkey]
        b_metrics["short_date"] = bkey[5:] if resolved_bucket == "daily" and len(bkey) == 10 else bkey[:7] if resolved_bucket == "monthly" else bkey[5:10]
        points.append(b_metrics)

    summary = compute_record_metrics(window_total_rec, lang_mode=lang_mode, lang_offsets=lang_offsets)
    summary["clarity_index"] = score_clarity_index(summary, baselines)

    # Evaluate >2.5σ bucket outliers across every longitudinal metric (and scrub when trim_outliers=True)
    detected_outliers_by_metric: dict[str, int] = {}
    trimmed_points_by_metric: dict[str, int] = {}
    outlier_bucket_keys: set[str] = set()

    for mkey, spec in METRIC_SPECS.items():
        series_vals = [
            p.get(mkey) for p in points
            if p.get(mkey) is not None and (not p.get("low_confidence") or mkey == "volume_words" or len(points) < 8)
        ]
        if len(series_vals) < 4:
            series_vals = [p.get(mkey) for p in points if p.get(mkey) is not None]
        low_cut, high_cut, _c, _sd = compute_outlier_cutoff(
            series_vals, baselines.get(mkey), mkey, eff_sigma
        )

        flagged_indices: list[int] = []
        for idx, pt in enumerate(points):
            v = pt.get(mkey)
            if v is None or not math.isfinite(float(v)):
                continue
            fv = float(v)
            is_runaway_pt = resolved_bucket == "daily" and str(pt.get("key")) in runaway_days
            is_stat_out = (high_cut is not None and fv > high_cut) or (low_cut is not None and fv < low_cut)
            if is_runaway_pt or is_stat_out:
                flagged_indices.append(idx)
                outlier_bucket_keys.add(str(pt.get("key", idx)))

        if flagged_indices:
            detected_outliers_by_metric[mkey] = len(flagged_indices)
            if trim_outliers and len(flagged_indices) < len(points):
                for idx in flagged_indices:
                    orig_v = points[idx][mkey]
                    points[idx].setdefault("_trimmed_values", {})[mkey] = orig_v
                    points[idx][mkey] = None
                trimmed_points_by_metric[mkey] = len(flagged_indices)

                # Recompute clean summary metric from remaining non-outlier buckets
                rem_pts = [p for p in points if p.get(mkey) is not None and math.isfinite(float(p[mkey]))]
                if rem_pts:
                    dec = int(spec.get("decimals", 2))
                    if mkey == "volume_words":
                        summary["volume_words"] = round(sum(float(p[mkey]) for p in rem_pts), 1)
                        summary["words"] = int(round(summary["volume_words"]))
                    else:
                        w_tot = sum(max(1.0, float(p.get("words", 1))) for p in rem_pts)
                        clean_avg = sum(float(p[mkey]) * max(1.0, float(p.get("words", 1))) for p in rem_pts) / max(w_tot, 1.0)
                        summary[mkey] = round(clean_avg, max(dec, 2))

    if trim_outliers and trimmed_points_by_metric:
        summary["clarity_index"] = score_clarity_index(summary, baselines)

    if resolved_bucket == "daily":
        span_days = max(0, (end_d - start_d).days) if (start_d and end_d) else 180
        roll_w = 7 if span_days <= 45 else 14
        roll_label = f"{roll_w}d trend"
    elif resolved_bucket == "weekly":
        roll_w = 4
        roll_label = "4w trend"
    else:
        roll_w = 3
        roll_label = "3m trend"

    rolling: dict[str, list[float | None]] = {}
    for mkey in METRIC_SPECS:
        rolling[mkey] = compute_rolling_series(points, mkey, roll_w)

    unique_outlier_buckets = (
        len(outlier_bucket_keys | set(runaway_days))
        if resolved_bucket == "daily"
        else len(outlier_bucket_keys) + len(runaway_days)
    )
    total_trimmed_points = sum(trimmed_points_by_metric.values()) + (
        trimmed_days_count if resolved_bucket != "daily" else 0
    )

    callout = generate_fogwatch_callout(points, summary, baselines, range_spec)
    if trim_outliers and (trimmed_days_count > 0 or trimmed_points_by_metric):
        n_tot_trim = (
            len(outlier_bucket_keys)
            if resolved_bucket == "daily"
            else trimmed_days_count + len(outlier_bucket_keys)
        )
        callout = f"{callout} [✂ >{eff_sigma:.1f}σ filter active: {max(1, n_tot_trim)} outlier bucket(s) excluded]"

    return {
        "range_spec": range_spec,
        "start_date": start_iso,
        "end_date": end_iso,
        "bucket_mode": resolved_bucket,
        "rolling_window": roll_w,
        "rolling_label": roll_label,
        "lang_mode": lang_mode,
        "trim_outliers": bool(trim_outliers),
        "outlier_sigma": eff_sigma,
        "trimmed_days_count": trimmed_days_count,
        "detected_runaway_days": len(runaway_days),
        "trimmed_points_by_metric": trimmed_points_by_metric,
        "detected_outliers_by_metric": detected_outliers_by_metric,
        "total_trimmed_points": total_trimmed_points,
        "total_outlier_buckets": unique_outlier_buckets,
        "points": points,
        "rolling": rolling,
        "summary": summary,
        "baselines": baselines,
        "baseline_days": baseline_days,
        "active_days_in_slice": active_days_in_slice,
        "callout": callout,
    }


def generate_fogwatch_callout(
    points: list[dict[str, Any]],
    summary: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    range_spec: Any,
) -> str:
    """Produce a concise, non-diagnostic longitudinal observation for the active slice."""
    words = int(summary.get("words", 0))
    if words < INSIGHTS_MIN_WORDS_CONFIDENT:
        return "Need at least ~40 words in this slice for stable lexical & pause trend signals."

    notes: list[str] = []
    ci = summary.get("clarity_index")
    if ci is not None:
        if ci >= 56.0:
            notes.append(f"Clarity Index {ci:.0f}/100 (clearer/more fluent than your baseline)")
        elif ci <= 44.0:
            notes.append(f"Clarity Index {ci:.0f}/100 (elevated pause/hesitation/retrieval load vs baseline)")
        else:
            notes.append(f"Clarity Index {ci:.0f}/100 (steady within your normal 40–60 band)")

    # First-half vs second-half trajectory drift on confident points
    conf_pts = [p for p in points if not p.get("low_confidence") and p.get("clarity_index") is not None]
    if len(conf_pts) >= 6:
        mid = len(conf_pts) // 2
        first_ci = statistics.mean(p["clarity_index"] for p in conf_pts[:mid])
        second_ci = statistics.mean(p["clarity_index"] for p in conf_pts[mid:])
        diff = second_ci - first_ci
        if abs(diff) >= 3.5:
            direction = "upward" if diff > 0 else "downward"
            notes.append(f"2nd-half trajectory shifted {direction} ({diff:+.1f} pts)")

    # Prioritize potential strain/fog excursions first, then positive diversity excursions
    strain_checks = (
        ("pause_long_pct", "long pauses >=5s", "%", "high"),
        ("vague_per_100", "vague placeholders", "/100w", "high"),
        ("fillers_per_100", "fillers", "/100w", "high"),
        ("mattr_diversity", "MATTR diversity", "%", "low"),
        ("late_night_pct", "late-night dictation", "%", "high"),
        ("mattr_diversity", "MATTR diversity", "%", "high"),
    )
    for mkey, label, unit, bound in strain_checks:
        val = summary.get(mkey)
        base = baselines.get(mkey)
        if val is None or not base:
            continue
        if bound == "high" and val > float(base["high"]):
            notes.append(f"{label} above usual ({val:.1f}{unit} vs {base['median']:.1f}{unit})")
            break
        if bound == "low" and val < float(base["low"]):
            notes.append(f"{label} below usual ({val:.1f}{unit} vs {base['median']:.1f}{unit})")
            break

    if not notes:
        return f"Analyzed {words:,} words across active slice."
    return " · ".join(notes) + "."


# ---------------------------------------------------------------------------
# Zero-LLM Deterministic Dimension Guide & Window Pattern Interpreter
# ---------------------------------------------------------------------------

def compute_series_correlation(
    points: list[dict[str, Any]],
    key_a: str,
    key_b: str,
) -> tuple[float | None, int]:
    """Compute Pearson ``r`` between two metric series across paired buckets."""
    pairs: list[tuple[float, float]] = []
    for p in points:
        va = p.get(key_a)
        vb = p.get(key_b)
        if va is None or vb is None:
            continue
        fa, fb = float(va), float(vb)
        if math.isfinite(fa) and math.isfinite(fb):
            pairs.append((fa, fb))
    n = len(pairs)
    if n < 5:
        return None, n
    ma = sum(x for x, _ in pairs) / n
    mb = sum(y for _, y in pairs) / n
    num = sum((x - ma) * (y - mb) for x, y in pairs)
    da = math.sqrt(sum((x - ma) ** 2 for x, _ in pairs))
    db = math.sqrt(sum((y - mb) ** 2 for _, y in pairs))
    if da <= 1e-9 or db <= 1e-9:
        return None, n
    return max(-1.0, min(1.0, num / (da * db))), n


def _z_of(summary: dict[str, Any], baselines: dict[str, dict[str, Any]], key: str) -> float | None:
    val = summary.get(key)
    base = baselines.get(key)
    if val is None or not base:
        return None
    f_val = float(val)
    f_med = float(base.get("median", 0.0))
    sigma = max(float(base.get("sigma", 1.0)), 1e-3)
    if not (math.isfinite(f_val) and math.isfinite(f_med) and math.isfinite(sigma)):
        return None
    return (f_val - f_med) / sigma


def _domain_implication(
    metric_key: str,
    z: float | None,
    traj_state: str,
    is_spiky: bool,
    summary: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> str:
    """Synthesize non-diagnostic cross-metric context for what the window pattern may mean."""
    z_vague = _z_of(summary, baselines, "vague_per_100") or 0.0
    z_pause = _z_of(summary, baselines, "pause_long_pct") or 0.0
    z_filler = _z_of(summary, baselines, "fillers_per_100") or 0.0
    z_mattr = _z_of(summary, baselines, "mattr_diversity") or 0.0
    z_vit = _z_of(summary, baselines, "vitality_balance") or 0.0
    z_night = _z_of(summary, baselines, "late_night_pct") or 0.0
    ci = summary.get("clarity_index")

    if metric_key == "clarity_index":
        if ci is not None and ci < 44.0:
            drivers = []
            if z_pause > 0.9:
                drivers.append("longer pauses")
            if z_vague > 0.9:
                drivers.append("more vague stand-in words")
            if z_mattr < -0.9:
                drivers.append("narrower active vocabulary")
            if z_filler > 0.9:
                drivers.append("elevated fillers")
            drv_txt = ", ".join(drivers[:3]) if drivers else "mild multi-factor hesitation shifts"
            night_bit = " coinciding with increased late-night dictation" if z_night > 1.0 else ""
            return (
                f"Driven mainly by {drv_txt}{night_bit} — sustained multi-day dips often reflect "
                f"sleep debt, cognitive overload, or mental fatigue rather than single-task noise."
            )
        if ci is not None and ci > 56.0:
            return (
                "Vocabulary breadth and retrieval continuity are crisp simultaneously — typical of "
                "rested, high-focus drafting or well-rehearsed subject matter."
            )
        if traj_state == "downward":
            return (
                "Overall average is still near baseline, but the downward second-half slope suggests "
                "accumulating fatigue or heavier task load worth watching."
            )
        return "Signals across vocabulary, specificity, and pause flow look balanced and consistent with your usual baseline."

    if metric_key in ("mattr_diversity", "content_density_pct", "lexical_long_word_percent"):
        if z is not None and z < -0.95:
            if z_vague > 0.8 or z_pause > 0.8 or z_filler > 0.8:
                return (
                    "Narrower word variety is paired with higher hesitation/placeholder load — a pattern "
                    "that can accompany word-finding fatigue or cognitive fog."
                )
            return (
                "Word variety is narrower while fluency stays clean — typically points to focused "
                "repetition on one technical topic or short command dictation rather than fatigue."
            )
        if z is not None and z > 0.95:
            return (
                "Broader, more varied word choice than usual — common during exploratory writing, "
                "planning across diverse topics, or high mental energy."
            )
        return "Vocabulary variety is holding steady — no sign of progressive lexical narrowing in this window."

    if metric_key in ("vague_per_100", "fillers_per_100", "disfluency_per_1k"):
        if z is not None and z > 0.95:
            if is_spiky:
                return (
                    "Concentrated in a few spiky sessions rather than every day — often caused by "
                    "thinking aloud on hard problems, unscripted brainstorming, or background noise."
                )
            if z_pause > 0.8 or z_mattr < -0.8:
                return (
                    "Sustained rise alongside slower pauses or narrower vocabulary — suggests higher "
                    "working-memory / word-retrieval effort (common with tiredness, stress, or multitasking)."
                )
            return (
                "Higher conversational hesitation without vocabulary loss — usually reflects informal "
                "improvised dictation rather than deeper retrieval strain."
            )
        if z is not None and z < -0.85:
            return "Markedly cleaner retrieval than usual — characteristic of clear focus or well-structured thoughts."
        return "Word-finding and verbal repair rates sit comfortably inside your normal personal range."

    if metric_key in ("pause_long_pct", "pause_p90_s", "pause_avg_s", "acoustic_silence_pct"):
        if z is not None and z > 0.95:
            if z_vague > 0.8 or z_filler > 0.8 or z_mattr < -0.8:
                return (
                    "Longer stalls co-occur with vague words or narrower vocabulary — a classic "
                    "cognitive-load / brain-fog signature where sentence planning takes extra effort."
                )
            return (
                "Longer gaps appear while vocabulary precision remains sharp — usually reflects "
                "deliberate thinking pauses (reading code/docs between utterances) or leaving the mic open."
            )
        if z is not None and z < -0.85:
            return "Brisk, continuous utterance flow with very few mid-thought stalls."
        return "Pause rhythm matches your normal speaking-and-thinking cadence."

    if metric_key in ("pace_words_entry", "words_per_sentence", "acoustic_wpm"):
        if z is not None and z < -0.95:
            if z_pause > 0.8 or z_vit < -0.8:
                return (
                    "Shorter, more fragmented utterances paired with slower pauses/strain markers — "
                    "can reflect low energy or cognitive fatigue."
            )
            return (
                "Shorter utterances with clean fluency — typically reflects quick chat replies, "
                "notes, or terminal/coding commands rather than long-form prose."
            )
        if z is not None and z > 0.95:
            return "Sustained multi-sentence dictation bursts — characteristic of deep drafting or structured explanation."
        return "Utterance length and elaboration match your typical working style."

    if metric_key == "self_focus_per_100":
        vit = summary.get("vitality_balance")
        if z is not None and z > 0.95:
            if vit is not None and vit < 45.0:
                return (
                    "Elevated 'I/ich' self-reference paired with strain/fatigue vocabulary — in "
                    "psycholinguistic studies, sustained co-elevation can accompany stress or rumination (non-diagnostic)."
                )
            return (
                "Higher 'I/ich' usage with steady/positive vitality — usually reflects personal "
                "planning, status updates, or reflective journaling rather than distress."
            )
        return "Balance between self-reference ('I/ich') and external/collective focus is in your normal band."

    if metric_key == "absolutist_per_100":
        if z is not None and z > 0.95:
            if z_vit < -0.8:
                return (
                    "More all-or-nothing phrasing ('always/never') alongside strain words — can appear "
                    "when feeling pressured, frustrated, or cognitively overloaded (non-diagnostic)."
                )
            return (
                "Slightly more categorical quantifiers without affect strain — often comes from "
                "specifying strict technical rules or requirements ('must', 'all', 'never')."
            )
        return " Nuanced, non-rigid phrasing within your personal baseline."

    if metric_key == "hedges_per_100":
        if z is not None and z > 0.95:
            return (
                "More tentative qualifiers ('maybe, perhaps, not sure') — common when weighing "
                "ambiguous decisions, exploring hypotheses, or feeling uncertain."
            )
        if z is not None and z < -0.85:
            return "Direct, decisive phrasing with minimal hedging."
        return "Uncertainty qualifiers sit within your usual conversational range."

    if metric_key == "vitality_balance":
        vit = summary.get("vitality_balance")
        if vit is not None and vit < 44.0:
            return (
                "Fatigue/stress words ('tired, foggy, overwhelmed / müde, erschöpft') outweigh vitality "
                "markers in this window — check alongside Clarity Index and Late-Night % for recovery cues."
            )
        if vit is not None and vit > 56.0:
            return "Positive energy, progress, and clarity words outweigh strain vocabulary in this slice."
        return "Affective word balance is neutral-to-steady around your personal baseline."

    if metric_key == "late_night_pct":
        ln = float(summary.get("late_night_pct") or 0.0)
        if ln >= 18.0 or (z is not None and z > 1.0):
            if ci is not None and ci < 46.0:
                return (
                    "Elevated 22:00–05:00 dictation coincides with a softer Clarity Index — late-night "
                    "schedule drift may be carrying over into cognitive sharpness."
                )
            return (
                "More late-night dictation than usual, though lexical clarity has held up so far — "
                "worth monitoring so multi-day sleep debt doesn't accumulate."
            )
        return "Very little late-night speech — activity is concentrated in daytime/evening hours."

    # volume_words fallback
    return "Reflects your overall dictation engagement across active days in the selected horizon."


def interpret_metric_window(
    slice_data: dict[str, Any] | None,
    metric_key: str,
    secondary_key: str | None = None,
) -> dict[str, str]:
    """Generate a 100% local, zero-LLM explanation + window pattern reading in <0.5ms."""
    spec = METRIC_SPECS.get(metric_key) or {
        "label": metric_key,
        "short": metric_key,
        "unit": "",
        "decimals": 1,
        "good_direction": "neutral",
        "description": "Local longitudinal speech indicator.",
    }
    explanation = str(spec.get("description", "")).strip()
    title = str(spec.get("label", metric_key))

    if not slice_data:
        return {
            "title": title,
            "explanation": explanation,
            "pattern": "Select a time horizon with recorded speech to evaluate your longitudinal pattern.",
            "correlation": "",
        }

    summary = slice_data.get("summary") or {}
    baselines = slice_data.get("baselines") or {}
    points = slice_data.get("points") or []
    words = int(summary.get("words", 0))

    if words == 0:
        return {
            "title": title,
            "explanation": explanation,
            "pattern": "No transcribed words in this time slice yet — switch to 6m, YTD, or All to view history.",
            "correlation": "",
        }

    val = summary.get(metric_key)
    unit = str(spec.get("unit", ""))
    dec = int(spec.get("decimals", 1))
    if val is None:
        if metric_key in ("acoustic_wpm", "acoustic_silence_pct"):
            pat = (
                "Recorded forward on new local dictations via ASR segment timestamps — make a new "
                "Local Live or Batch recording to start populating this curve."
            )
        else:
            pat = "Not enough data points on this dimension in the selected slice yet."
        return {"title": title, "explanation": explanation, "pattern": pat, "correlation": ""}

    if words < INSIGHTS_MIN_WORDS_CONFIDENT and metric_key != "volume_words":
        return {
            "title": title,
            "explanation": explanation,
            "pattern": (
                f"Light sample ({words} words across slice; current {float(val):.{dec}f}{unit}). "
                f"Dictate ~40+ words or widen the horizon for a reliable baseline & trend reading."
            ),
            "correlation": "",
        }

    # 1. Level vs. 90-day Personal Baseline
    base = baselines.get(metric_key)
    z = _z_of(summary, baselines, metric_key)
    f_val = float(val)
    if base and z is not None:
        b_med = float(base.get("median", 0.0))
        norm_tag = "90d personal norm" if base.get("is_personal") else "reference norm"
        if abs(z) <= 0.85:
            lvl_bit = f"Within your {norm_tag} ({f_val:.{dec}f}{unit} vs {b_med:.{dec}f}{unit})"
        elif abs(z) <= 1.65:
            dir_w = "moderately above" if z > 0 else "moderately below"
            lvl_bit = f"{dir_w.capitalize()} your {norm_tag} ({f_val:.{dec}f}{unit} vs {b_med:.{dec}f}{unit}, {z:+.1f}σ)"
        else:
            dir_w = "clearly above" if z > 0 else "clearly below"
            lvl_bit = f"{dir_w.capitalize()} your {norm_tag} ({f_val:.{dec}f}{unit} vs {b_med:.{dec}f}{unit}, {z:+.1f}σ)"
    else:
        lvl_bit = f"Window total/mean: {f_val:.{dec}f}{unit} across {slice_data.get('active_days_in_slice', 0)} active day(s)"

    # 2. Trajectory & Volatility across buckets
    valid_pts = [
        p for p in points
        if p.get(metric_key) is not None
        and math.isfinite(float(p[metric_key]))
        and (not p.get("low_confidence") or metric_key == "volume_words" or len(points) < 6)
    ]
    traj_state = "flat"
    traj_bit = ""
    is_spiky = False

    if len(valid_pts) >= 4:
        p_vals = [float(p[metric_key]) for p in valid_pts]
        scale = max(float(base["sigma"]) if base else 0.0, statistics.pstdev(p_vals) or 1.0, 1e-3)
        mid = len(p_vals) // 2
        m1 = statistics.mean(p_vals[:mid])
        m2 = statistics.mean(p_vals[mid:])
        half_z = (m2 - m1) / scale

        if half_z >= 0.55:
            traj_state = "upward"
            traj_bit = f"trending upward in the 2nd half ({m1:.{dec}f} → {m2:.{dec}f}{unit})"
        elif half_z <= -0.55:
            traj_state = "downward"
            traj_bit = f"trending downward in the 2nd half ({m1:.{dec}f} → {m2:.{dec}f}{unit})"
        else:
            # Check recent 2-point tail vs earlier window mean
            if len(p_vals) >= 6:
                early_m = statistics.mean(p_vals[:-2])
                tail_m = statistics.mean(p_vals[-2:])
                tail_z = (tail_m - early_m) / scale
                if abs(tail_z) >= 0.85:
                    t_dir = "uptick" if tail_z > 0 else "dip"
                    traj_bit = f"mostly flat with a recent {t_dir} ({tail_m:.{dec}f}{unit})"
                else:
                    traj_bit = "steady across the window"
            else:
                traj_bit = "steady across the window"

        # Check episodic spikiness vs sustained level
        if base:
            b_hi = float(base.get("high", float("inf")))
            b_lo = float(base.get("low", float("-inf")))
            excursions = sum(1 for x in p_vals if x > b_hi or x < b_lo)
            if 1 <= excursions <= max(2, len(p_vals) // 5):
                is_spiky = True
                traj_bit += f" ({excursions}/{len(p_vals)} buckets outside band)"
    elif len(valid_pts) in (2, 3):
        traj_bit = f"{len(valid_pts)} buckets in view (widen horizon for trend slope)"
    else:
        traj_bit = "single bucket in view"

    # 3. Domain synthesis (cross-signal corroboration, zero LLM calls)
    meaning = _domain_implication(metric_key, z, traj_state, is_spiky, summary, baselines)

    # 4. Outlier status note
    trim_on = bool(slice_data.get("trim_outliers"))
    trimmed_n = int((slice_data.get("trimmed_points_by_metric") or {}).get(metric_key, 0))
    detected_n = int((slice_data.get("detected_outliers_by_metric") or {}).get(metric_key, 0))
    sigma_lbl = float(slice_data.get("outlier_sigma", INSIGHTS_OUTLIER_SIGMA))
    outlier_note = ""
    if trim_on and trimmed_n > 0:
        outlier_note = f" [✂ {trimmed_n} >{sigma_lbl:.1f}σ spike(s) excluded]"
    elif (not trim_on) and detected_n > 0:
        outlier_note = f" [Tip: {detected_n} >{sigma_lbl:.1f}σ spike(s) present — click '✂ >2.5σ' to filter accidental pushes]"

    # 5. Optional Primary vs. Secondary Overlay Correlation
    corr_txt = ""
    if secondary_key and secondary_key != metric_key and secondary_key in METRIC_SPECS:
        r_val, n_pair = compute_series_correlation(points, metric_key, secondary_key)
        sec_short = METRIC_SPECS[secondary_key].get("short", secondary_key)
        if r_val is not None:
            if r_val >= 0.35:
                rel = "co-move positively"
            elif r_val <= -0.35:
                rel = "move inversely"
            else:
                rel = "vary mostly independently"
            corr_txt = f"Overlay vs. {sec_short}: r = {r_val:+.2f} across {n_pair} buckets ({rel})."
        elif n_pair >= 5:
            corr_txt = f"Overlay vs. {sec_short}: r = +0.00 across {n_pair} buckets (one series is flat/constant)."

    lead = f"{lvl_bit}, {traj_bit}." if traj_bit else f"{lvl_bit}."
    pattern_full = f"{lead} {meaning}{outlier_note}".strip()
    return {
        "title": title,
        "explanation": explanation,
        "pattern": pattern_full,
        "correlation": corr_txt,
    }


def interpret_categorical_view(
    view_kind: str,
    context: dict[str, Any] | None = None,
    slice_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Zero-LLM explanation & pattern takeaway for categorical/breakdown/radar/heatmap views."""
    ctx = context or {}
    summ = (slice_data or {}).get("summary") or {}

    if view_kind == "pause_breakdown":
        s_cnt = int(ctx.get("short", 0))
        m_cnt = int(ctx.get("medium", 0))
        l_cnt = int(ctx.get("long", 0))
        tot = max(1, s_cnt + m_cnt + l_cnt)
        l_pct = (l_cnt * 100.0) / tot
        expl = (
            "Splits all within-burst dictation gaps (<=120s) into brisk transitions (<2s), "
            "clause-planning pauses (2–5s), and extended stalls (>=5s)."
        )
        if s_cnt + m_cnt + l_cnt == 0:
            pat = "Need consecutive utterances within 120s to populate pause buckets."
        elif l_pct >= 28.0:
            pat = (
                f"Extended stalls (>=5s) make up {l_pct:.0f}% of transitions — check whether this "
                f"coincides with vague words (retrieval load) or deliberate reading/coding pauses."
            )
        elif (s_cnt * 100.0 / tot) >= 60.0:
            pat = f"Brisk <2s transitions dominate ({s_cnt}/{tot}) — fluid, continuous dictation momentum."
        else:
            pat = f"Balanced rhythm across brisk ({s_cnt}), medium ({m_cnt}), and long ({l_cnt}) pauses."
        return {"title": "Pause Duration Breakdown", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "clarity_breakdown":
        top_items = ctx.get("top_items") or []
        expl = (
            "Ranks your most frequent vocalized hesitation and discourse-filler habits across "
            "English and German in the selected window."
        )
        if not top_items:
            pat = "Zero filler habits detected in this slice — remarkably clean phrasing."
        else:
            lead_k, lead_n = top_items[0]
            pat = (
                f"'{lead_k}' is your dominant hesitation habit ({lead_n}x in slice). "
                f"Acoustic 'um/uh/äh/ähm' spikes often track speech-planning effort, whereas 'like/halt/quasi' reflect conversational style."
            )
        return {"title": "Top Filler Habits", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "lexical_breakdown":
        top_words = ctx.get("top_words") or []
        mattr = summ.get("mattr_diversity")
        expl = (
            "Highlights your highest-frequency substantive content words (excluding EN/DE stopwords) "
            "to surface recurring mental themes and active projects."
        )
        if not top_words:
            pat = "Dictate more text in this window to surface recurring content themes."
        else:
            names = ", ".join(f"'{w}' ({c})" for w, c in top_words[:4])
            m_bit = f" with {mattr:.1f}% MATTR-50 diversity" if mattr is not None else ""
            pat = f"Dominant topical anchors in this window: {names}{m_bit}."
        return {"title": "Core Content Themes", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "interventions":
        oq = int(ctx.get("open_q", 0))
        rf = int(ctx.get("reflect", 0))
        af = int(ctx.get("affirm", 0))
        su = int(ctx.get("summary", 0))
        sc = int(ctx.get("self_count", 0))
        wc = int(ctx.get("we_count", 0))
        expl = (
            "Counts conversational stance markers — open questions, empathetic reflections, "
            "affirmations, summaries, and individual ('I/ich') vs. collective ('we/wir') framing."
        )
        stance = "collaborative ('we/wir'-leaning)" if wc > sc else "self-directed ('I/ich'-leaning)"
        if oq + rf + af + su == 0:
            pat = f"Primarily direct declarative dictation ({stance}, {sc}:{wc} I-to-We ratio) with few explicit Q&A/summary cues."
        elif oq > (rf + af + su):
            pat = f"Inquiry-heavy window ({oq} open questions; {stance}) — typical of exploratory problem-solving or prompting."
        else:
            pat = f"Structured communicative mix (Open Q {oq}, Affirm {af}, Summary {su}; {stance})."
        return {"title": "Conversational Stance & Mix", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "activity":
        mtype = str(ctx.get("metric_type", "Words (Volume)"))
        n_days = int(ctx.get("active_days", 0))
        mean_v = ctx.get("mean_val")
        expl = (
            "Calendar heatmap over up to 52 weeks (Mon–Sun rows) revealing weekday habits, "
            "consistency streaks, and clusters of high clarity or strain."
        )
        if n_days == 0:
            pat = "No active calendar days recorded for this heatmap layer yet."
        else:
            m_str = f" (mean {float(mean_v):.1f})" if mean_v is not None else ""
            pat = f"Spans {n_days} active day(s) on '{mtype}'{m_str} — hover individual squares to spot multi-day clusters vs. isolated off-days."
        return {"title": f"Calendar Heatmap · {mtype}", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "profile":
        axes = ctx.get("axes") or {}
        expl = (
            "5-axis fingerprint balancing Vocabulary Variety (MATTR-50), Cognitive Clarity, "
            "Lexical Vitality, Inquiry Drive (questions), and Sustained Thought (sentence length)."
        )
        if not axes:
            pat = "Need at least 10 words in the selected window to build your 5-axis profile."
        else:
            strongest = max(axes.items(), key=lambda kv: kv[1])
            weakest = min(axes.items(), key=lambda kv: kv[1])
            pat = (
                f"Strongest dimension in this window: {strongest[0]} ({strongest[1]:.0f}/100); "
                f"softest axis: {weakest[0]} ({weakest[1]:.0f}/100)."
            )
        return {"title": "5-Axis Speaker Fingerprint", "explanation": expl, "pattern": pat, "correlation": ""}

    if view_kind == "mood":
        scores = ctx.get("scores") or []
        vit = summ.get("vitality_balance")
        expl = (
            "Optional AI-crunched daily emotional tone (0–100, 50 = neutral). For 100% offline "
            "zero-LLM affect tracking, check the 'Mind → Vitality' tab anytime."
        )
        if not scores:
            v_hint = f" (Offline Mind Vitality in this window is {vit:.0f}/100.)" if vit is not None else ""
            pat = f"No cached AI mood scores in this range yet.{v_hint}"
        else:
            avg_m = sum(scores) / len(scores)
            tone = "upbeat/positive" if avg_m >= 58 else "strained/low" if avg_m <= 42 else "steady/balanced"
            pat = f"Mean AI mood across {len(scores)} crunched day(s) is {avg_m:.0f}/100 ({tone})."
        return {"title": "Mood Trajectory", "explanation": expl, "pattern": pat, "correlation": ""}

    return {"title": view_kind, "explanation": "", "pattern": "", "correlation": ""}

