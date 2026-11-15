"""Speech Insights, bilingual (EN/DE) lexical & filler analytics, histograms, and AI Daily Brief engine.

Pure analytics functions at the top of this module have zero X11/Tkinter runtime requirements
and can be imported and unit-tested in headless environments. ``InsightsMixin`` provides all
10 Speech Insights tabs, radar/heatmap/Gantt/sparkline Canvas visualizations, and the
Today / All-Time histograms using 100% ``CustomTkinter`` controls (zero ``tk.Button`` or
``tk.OptionMenu`` widgets).
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
import threading
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import paths
from .config import (
    ENGLISH_FILLER_PATTERNS,
    ENGLISH_STOPWORDS,
    GERMAN_DETECT_THRESHOLD,
    GERMAN_FILLER_PATTERNS,
    GERMAN_FUNCTION_WORDS,
    GERMAN_STOPWORDS,
    INSIGHTS_RANGE_DEFAULT,
    INSIGHTS_RANGE_OPTIONS,
    PEER_BENCHMARK_NOTE,
    PEER_BENCHMARKS,
    PEER_PROFILE,
    PERSONAL_BASELINE_BAND_FRACTION,
    PERSONAL_BASELINE_DAYS,
    PERSONAL_BASELINE_MIN_DAYS,
    THEME_COLORS,
    TYPING_WPM,
    mono_font,
    ui_font,
)

log = logging.getLogger("neurowhisper.insights")

_WORD_RE = re.compile(r"\b[^\W\d_]+(?:'[^\W\d_]+)*\b", re.UNICODE)
_FULL_TS_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
_SHORT_TS_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
_DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

OPEN_QUESTION_STARTERS_EN = (
    "what", "how", "why", "where", "when", "who", "which", "tell me", "walk me through",
)
OPEN_QUESTION_STARTERS_DE = (
    "was", "wie", "warum", "weshalb", "wieso", "wo", "wann", "wer", "welche", "erzähl",
)
REFLECTION_PATTERNS = re.compile(
    r"\b(it sounds like|you feel|you're feeling|what i hear|so you|es klingt|du fühlst|ich höre|du meinst)\b",
    re.IGNORECASE,
)
AFFIRMATION_PATTERNS = re.compile(
    r"\b(great job|well done|appreciate|thank you|proud of|strong point|super gemacht|danke|wertschätze|klasse|stark)\b",
    re.IGNORECASE,
)
SUMMARY_PATTERNS = re.compile(
    r"\b(to summarize|in summary|overall|let's recap|key takeaway|zusammenfassend|insgesamt|kurz gesagt|fazit)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Headless-Safe Pure Analytics & Archive Parsing Helpers
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    """Extract lowercase Unicode alphabetic tokens (preserving internal apostrophes and umlauts)."""
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def detect_entry_language(words: List[str]) -> str:
    """Heuristically classify tokenized speech as ``'de'`` or ``'en'``."""
    if not words:
        return "en"
    de_hits = sum(1 for w in words if w in GERMAN_FUNCTION_WORDS)
    if (de_hits / float(len(words))) >= GERMAN_DETECT_THRESHOLD:
        return "de"
    if any(ch in "".join(words) for ch in "äöüß"):
        return "de"
    return "en"


def filler_patterns_for(lang: str) -> Dict[str, str]:
    """Return compiled-ready regex filler map for ``'de'`` or ``'en'``."""
    if (lang or "").lower().startswith("de"):
        return dict(GERMAN_FILLER_PATTERNS)
    return dict(ENGLISH_FILLER_PATTERNS)


def stopwords_for(lang: str) -> Set[str]:
    """Return stopword set for ``'de'``, ``'en'``, or ``'mixed'``."""
    code = (lang or "en").lower()
    if code.startswith("de"):
        return set(GERMAN_STOPWORDS)
    if code == "mixed":
        return set(ENGLISH_STOPWORDS) | set(GERMAN_STOPWORDS)
    return set(ENGLISH_STOPWORDS)


def count_fillers(text: str, lang: str = "en") -> Tuple[int, Dict[str, int]]:
    """Count filler occurrences in ``text`` for ``lang``, returning ``(total, per_category)``."""
    if not text:
        return 0, {}
    patterns = filler_patterns_for(lang)
    breakdown: Dict[str, int] = {}
    total = 0
    lowered = text.lower()
    for label, pattern in patterns.items():
        hits = len(re.findall(pattern, lowered, flags=re.IGNORECASE | re.UNICODE))
        if hits > 0:
            breakdown[label] = hits
            total += hits
    return total, breakdown


def count_immediate_repetitions(words: List[str]) -> int:
    """Count immediate consecutive word repetitions (e.g., ``'the the'``, ``'ich ich'``)."""
    if len(words) < 2:
        return 0
    reps = 0
    for idx in range(1, len(words)):
        if len(words[idx]) >= 2 and words[idx] == words[idx - 1]:
            reps += 1
    return reps


def parse_transcript_file(
    filepath: Path | str,
    fallback_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse a ``neurowhisper`` archive file (``YYYY-MM.<HOST>.txt`` or ``YYYY-MM-DD.txt``).

    Supports both ``[YYYY-MM-DD HH:MM:SS] text`` and legacy ``[HH:MM:SS] text`` lines,
    and appends continuation paragraphs to the preceding timestamped entry.
    """
    path = Path(filepath)
    if not path.is_file():
        return []
    if "conflicted copy" in path.name.lower():
        return []

    if fallback_date is None:
        m_date = _DATE_IN_FILENAME_RE.search(path.stem)
        if m_date:
            fallback_date = m_date.group(1)

    entries: List[Dict[str, Any]] = []
    current_dt: Optional[datetime] = None
    current_lines: List[str] = []

    def _flush() -> None:
        nonlocal current_dt, current_lines
        if current_dt is None:
            current_lines = []
            return
        body = "\n".join(line for line in current_lines if line.strip()).strip()
        current_lines = []
        if not body:
            return
        words = tokenize_words(body)
        if not words:
            return
        lang = detect_entry_language(words)
        filler_total, filler_map = count_fillers(body, lang)
        entries.append(
            {
                "timestamp": current_dt,
                "date": current_dt.strftime("%Y-%m-%d"),
                "hour": current_dt.hour,
                "text": body,
                "words": words,
                "word_count": len(words),
                "char_count": len(body),
                "lang": lang,
                "filler_count": filler_total,
                "filler_breakdown": filler_map,
                "repetitions": count_immediate_repetitions(words),
                "source_file": path.name,
            }
        )

    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for raw_line in raw_text.splitlines():
        m_full = _FULL_TS_LINE_RE.match(raw_line)
        if m_full:
            _flush()
            day_str, time_str, content = m_full.groups()
            try:
                current_dt = datetime.strptime(f"{day_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                current_dt = None
            if content.strip():
                current_lines.append(content.strip())
            continue

        m_short = _SHORT_TS_LINE_RE.match(raw_line)
        if m_short and fallback_date:
            _flush()
            time_str, content = m_short.groups()
            try:
                current_dt = datetime.strptime(f"{fallback_date} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                current_dt = None
            if content.strip():
                current_lines.append(content.strip())
            continue

        if current_dt is not None and raw_line.strip():
            current_lines.append(raw_line.strip())

    _flush()
    return entries


def load_all_transcript_entries(transcriptions_dir: Path | str) -> List[Dict[str, Any]]:
    """Load and chronologically sort all transcript entries across peer archive files."""
    folder = Path(transcriptions_dir)
    if not folder.is_dir():
        return []

    all_entries: List[Dict[str, Any]] = []
    for txt_file in sorted(folder.glob("*.txt")):
        if "conflicted copy" in txt_file.name.lower():
            continue
        all_entries.extend(parse_transcript_file(txt_file))

    all_entries.sort(key=lambda item: item["timestamp"])
    return all_entries


def filter_entries_by_days(
    entries: Iterable[Dict[str, Any]],
    days: Optional[int],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Filter parsed transcript entries to the requested window (`0`=Today, `7`, `30`, `None`=All)."""
    items = list(entries)
    if days is None:
        return items
    ref = now or datetime.now()
    if days == 0:
        today_str = ref.strftime("%Y-%m-%d")
        return [e for e in items if e.get("date") == today_str]
    cutoff = (ref - timedelta(days=int(days))).date()
    return [e for e in items if e["timestamp"].date() >= cutoff]


def _windowed_ttr(words: List[str], window: int = 100) -> float:
    """Compute Moving-Average Type-Token Ratio (MATTR) as a percentage ``0..100``."""
    if not words:
        return 0.0
    if len(words) <= window:
        return (len(set(words)) / float(len(words))) * 100.0
    ratios: List[float] = []
    step = max(10, window // 4)
    for start in range(0, len(words) - window + 1, step):
        chunk = words[start : start + window]
        ratios.append(len(set(chunk)) / float(window))
    return (sum(ratios) / float(len(ratios))) * 100.0 if ratios else 0.0


def compute_speech_insights(
    entries: List[Dict[str, Any]],
    all_entries_for_baseline: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive bilingual Speech Insights metrics for ``entries``."""
    if not entries:
        return {
            "empty": True,
            "entry_count": 0,
            "total_words": 0,
            "active_days": 0,
            "dominant_lang": "en",
        }

    total_entries = len(entries)
    total_words = sum(e["word_count"] for e in entries)
    active_dates = sorted({e["date"] for e in entries})
    active_days = len(active_dates)

    # Language mix
    de_entries = sum(1 for e in entries if e.get("lang") == "de")
    en_entries = total_entries - de_entries
    if de_entries > 0 and en_entries > 0:
        dominant_lang = "mixed"
    elif de_entries > en_entries:
        dominant_lang = "de"
    else:
        dominant_lang = "en"

    # Pace & Stability
    words_per_entry = [float(e["word_count"]) for e in entries]
    avg_words_entry = sum(words_per_entry) / float(total_entries)
    if len(words_per_entry) >= 2 and avg_words_entry > 0:
        stdev_w = statistics. pstdev(words_per_entry)
        cv = stdev_w / avg_words_entry
        pace_stability = max(0.0, min(100.0, 100.0 * (1.0 - min(cv, 1.25) / 1.25)))
    else:
        pace_stability = 85.0

    # Utterance length buckets
    length_buckets = {"1–10w": 0, "11–25w": 0, "26–60w": 0, "61+w": 0}
    for wc in words_per_entry:
        if wc <= 10:
            length_buckets["1–10w"] += 1
        elif wc <= 25:
            length_buckets["11–25w"] += 1
        elif wc <= 60:
            length_buckets["26–60w"] += 1
        else:
            length_buckets["61+w"] += 1

    # Inter-utterance pauses (within same session <= 90s gap)
    session_gaps: List[float] = []
    for idx in range(1, total_entries):
        delta = (entries[idx]["timestamp"] - entries[idx - 1]["timestamp"]).total_seconds()
        # Subtract estimated speaking time of entry idx (~150 wpm = 2.5 words/sec)
        est_speak = entries[idx]["word_count"] / 2.5
        gap = max(0.2, delta - est_speak)
        if 0.2 <= delta <= 90.0:
            session_gaps.append(min(gap, delta))

    avg_pause_sec = (sum(session_gaps) / float(len(session_gaps))) if session_gaps else 2.2
    long_pauses = sum(1 for g in session_gaps if g >= 6.0)
    long_pause_share = (long_pauses / float(len(session_gaps))) if session_gaps else 0.0
    pause_buckets = {"<1.5s": 0, "1.5–4s": 0, "4–10s": 0, "10s+": 0}
    for g in session_gaps:
        if g < 1.5:
            pause_buckets["<1.5s"] += 1
        elif g < 4.0:
            pause_buckets["1.5–4s"] += 1
        elif g < 10.0:
            pause_buckets["4–10s"] += 1
        else:
            pause_buckets["10s+"] += 1

    # Clarity: Fillers & Repetitions
    total_fillers = sum(int(e.get("filler_count", 0)) for e in entries)
    filler_breakdown: Counter = Counter()
    for e in entries:
        for k, v in (e.get("filler_breakdown") or {}).items():
            filler_breakdown[k] += int(v)
    fillers_per_100 = (total_fillers / float(max(1, total_words))) * 100.0
    total_reps = sum(int(e.get("repetitions", 0)) for e in entries)
    reps_per_1k = (total_reps / float(max(1, total_words))) * 1000.0

    # Lexical diversity & Top vocabulary
    flat_words: List[str] = []
    for e in entries:
        flat_words.extend(e["words"])
    lexical_diversity = _windowed_ttr(flat_words, window=100)
    long_words = sum(1 for w in flat_words if len(w) >= 7)
    long_word_pct = (long_words / float(max(1, len(flat_words)))) * 100.0

    stops = stopwords_for("mixed" if dominant_lang == "mixed" else dominant_lang)
    vocab_counter = Counter(w for w in flat_words if len(w) >= 4 and w not in stops)
    top_vocab = vocab_counter.most_common(12)

    # Coaching / Communication Interventions
    sentences_total = 0
    open_questions = 0
    closed_questions = 0
    reflections = 0
    affirmations = 0
    summaries = 0
    for e in entries:
        txt = e["text"]
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
        sentences_total += max(1, len(parts))
        for s in parts:
            s_low = s.lower().lstrip("\"'“„ ")
            if s.endswith("?"):
                if s_low.startswith(OPEN_QUESTION_STARTERS_EN) or s_low.startswith(OPEN_QUESTION_STARTERS_DE):
                    open_questions += 1
                else:
                    closed_questions += 1
        reflections += len(REFLECTION_PATTERNS.findall(txt))
        affirmations += len(AFFIRMATION_PATTERNS.findall(txt))
        summaries += len(SUMMARY_PATTERNS.findall(txt))

    denom_sent = float(max(1, sentences_total))
    open_q_per_100 = (open_questions / denom_sent) * 100.0
    total_q = open_questions + closed_questions
    reflection_ratio = float(reflections) / float(max(1, total_q))
    affirmations_per_100 = (affirmations / denom_sent) * 100.0
    summaries_per_100 = (summaries / denom_sent) * 100.0

    # Daily & Hourly Activity Aggregations
    daily_words: Dict[str, int] = defaultdict(int)
    daily_sessions: Dict[str, int] = defaultdict(int)
    daily_fillers: Dict[str, int] = defaultdict(int)
    hourly_words: Dict[int, int] = defaultdict(int)
    for e in entries:
        d_key = e["date"]
        daily_words[d_key] += e["word_count"]
        daily_sessions[d_key] += 1
        daily_fillers[d_key] += int(e.get("filler_count", 0))
        hourly_words[int(e["hour"])] += e["word_count"]

    time_saved_minutes = total_words / float(TYPING_WPM)

    # Radar normalized 0..100 scores
    clarity_score = max(10.0, min(100.0, 100.0 - fillers_per_100 * 14.0 - reps_per_1k * 1.5))
    lexical_score = max(10.0, min(100.0, (lexical_diversity / 52.0) * 100.0))
    fluency_score = max(10.0, min(100.0, 100.0 - long_pause_share * 140.0))
    inquiry_score = max(15.0, min(100.0, 40.0 + open_q_per_100 * 3.0 + reflection_ratio * 15.0))
    radar_axes = [
        ("Clarity", round(clarity_score, 1)),
        ("Lexical", round(lexical_score, 1)),
        ("Stability", round(pace_stability, 1)),
        ("Fluency", round(fluency_score, 1)),
        ("Inquiry", round(inquiry_score, 1)),
    ]

    # Personal 90-day baseline check
    baseline_source = all_entries_for_baseline if all_entries_for_baseline is not None else entries
    recent_90 = filter_entries_by_days(baseline_source, PERSONAL_BASELINE_DAYS)
    baseline_days = len({e["date"] for e in recent_90})
    has_personal_baseline = baseline_days >= PERSONAL_BASELINE_MIN_DAYS
    baseline_metrics: Dict[str, Tuple[float, float]] = {}
    if has_personal_baseline and recent_90:
        b_words = sum(e["word_count"] for e in recent_90)
        b_fillers_100 = (sum(e.get("filler_count", 0) for e in recent_90) / float(max(1, b_words))) * 100.0
        b_avg_w = b_words / float(len(recent_90))
        frac = PERSONAL_BASELINE_BAND_FRACTION
        baseline_metrics = {
            "pace_avg_words_entry": (b_avg_w * (1.0 - frac), b_avg_w * (1.0 + frac)),
            "clarity_fillers_per_100": (max(0.0, b_fillers_100 * (1.0 - frac)), b_fillers_100 * (1.0 + frac) + 0.3),
        }

    return {
        "empty": False,
        "entry_count": total_entries,
        "total_words": total_words,
        "active_days": active_days,
        "time_saved_minutes": time_saved_minutes,
        "dominant_lang": dominant_lang,
        "de_entries": de_entries,
        "en_entries": en_entries,
        "avg_words_entry": avg_words_entry,
        "pace_stability": pace_stability,
        "length_buckets": length_buckets,
        "avg_pause_sec": avg_pause_sec,
        "long_pause_share": long_pause_share,
        "pause_buckets": pause_buckets,
        "pause_sample_count": len(session_gaps),
        "total_fillers": total_fillers,
        "fillers_per_100": fillers_per_100,
        "filler_breakdown": dict(filler_breakdown.most_common(8)),
        "total_reps": total_reps,
        "reps_per_1k": reps_per_1k,
        "lexical_diversity": lexical_diversity,
        "long_word_pct": long_word_pct,
        "top_vocab": top_vocab,
        "open_q_per_100": open_q_per_100,
        "reflection_ratio": reflection_ratio,
        "affirmations_per_100": affirmations_per_100,
        "summaries_per_100": summaries_per_100,
        "sentences_total": sentences_total,
        "daily_words": dict(daily_words),
        "daily_sessions": dict(daily_sessions),
        "daily_fillers": dict(daily_fillers),
        "hourly_words": dict(hourly_words),
        "radar_axes": radar_axes,
        "has_personal_baseline": has_personal_baseline,
        "baseline_days": baseline_days,
        "baseline_metrics": baseline_metrics,
    }


def _benchmark_status(metric_key: str, value: float, personal_bands: Dict[str, Tuple[float, float]]) -> str:
    if metric_key in personal_bands:
        low, high = personal_bands[metric_key]
        src = "personal"
    else:
        band = PEER_BENCHMARKS.get(metric_key)
        if not band:
            return ""
        low, high = band["low"], band["high"]
        src = "heuristic"

    if value < low:
        return f"Below {src} band ({low:.1f}–{high:.1f})"
    if value > high:
        return f"Above {src} band ({low:.1f}–{high:.1f})"
    return f"Within {src} band ({low:.1f}–{high:.1f})"


# ---------------------------------------------------------------------------
# InsightsMixin for WhisperApp (100% CustomTkinter Buttons & OptionMenus)
# ---------------------------------------------------------------------------

class InsightsMixin:
    """UI mixin providing Today/All-Time Histograms and all 10 Speech Insights tabs."""

    INSIGHT_TABS: Tuple[str, ...] = (
        "Summary",
        "AI Brief",
        "Trends",
        "Interventions",
        "Pace",
        "Pause",
        "Clarity",
        "Lexical",
        "Activity",
        "Notes",
    )

    def _init_insights_state(self) -> None:
        self.histogram_mode: str = "today"
        self.insights_range_label: str = str(
            self.config.get("insights_range", INSIGHTS_RANGE_DEFAULT)
        )
        if self.insights_range_label not in INSIGHTS_RANGE_OPTIONS:
            self.insights_range_label = INSIGHTS_RANGE_DEFAULT
        self.active_insight_tab: str = "Summary"
        self._cached_insights: Optional[Dict[str, Any]] = None
        self._ai_insights_data: Dict[str, Dict[str, Any]] = self._load_ai_insights_disk()
        self._ai_crunch_running: bool = False
        self._ai_timeline_days: str = "14 days"
        self._ai_mood_days: str = "14 days"
        self._activity_metric: str = "Words"

    def _load_ai_insights_disk(self) -> Dict[str, Dict[str, Any]]:
        ai_file = paths.get_ai_insights_file()
        if not ai_file.is_file():
            return {}
        try:
            with open(ai_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save_ai_insights_disk(self) -> None:
        try:
            paths.write_json_atomic(paths.get_ai_insights_file(), self._ai_insights_data, indent=2)
        except Exception as exc:
            log.warning("Failed to save ai_insights.json: %s", exc)

    # ------------------------------------------------------------------
    # Histograms Section (Today Hourly vs. All-Time Daily)
    # ------------------------------------------------------------------

    def build_histograms_content(self, parent) -> None:
        import tkinter as tk
        import customtkinter as ctk

        toggle_row = tk.Frame(parent, bg=self.colors["accent_bg"])
        toggle_row.pack(fill="x", pady=(2, 6))

        self.btn_hist_today = ctk.CTkButton(
            toggle_row,
            text="Today (Hourly)",
            height=26,
            corner_radius=6,
            font=ui_font(11, "bold"),
            fg_color=self.colors["teal_dim"],
            hover_color=self.colors["teal"],
            text_color=self.colors["fg"],
            command=lambda: self.switch_histogram_view("today"),
        )
        self.btn_hist_today.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_hist_alltime = ctk.CTkButton(
            toggle_row,
            text="All-Time (14d)",
            height=26,
            corner_radius=6,
            font=ui_font(11),
            fg_color=self.colors["bg"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_muted"],
            command=lambda: self.switch_histogram_view("alltime"),
        )
        self.btn_hist_alltime.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.histogram_canvas = tk.Canvas(
            parent,
            height=115,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        self.histogram_canvas.pack(fill="x", pady=(2, 2))
        self.histogram_canvas.bind("<Configure>", lambda _e: self.draw_histogram())

    def switch_histogram_view(self, mode: str) -> None:
        self.histogram_mode = mode
        if hasattr(self, "btn_hist_today") and hasattr(self, "btn_hist_alltime"):
            if mode == "today":
                self.btn_hist_today.configure(
                    fg_color=self.colors["teal_dim"],
                    text_color=self.colors["fg"],
                    font=ui_font(11, "bold"),
                )
                self.btn_hist_alltime.configure(
                    fg_color=self.colors["bg"],
                    text_color=self.colors["text_muted"],
                    font=ui_font(11),
                )
            else:
                self.btn_hist_alltime.configure(
                    fg_color=self.colors["teal_dim"],
                    text_color=self.colors["fg"],
                    font=ui_font(11, "bold"),
                )
                self.btn_hist_today.configure(
                    fg_color=self.colors["bg"],
                    text_color=self.colors["text_muted"],
                    font=ui_font(11),
                )
        self.draw_histogram()

    def draw_histogram(self) -> None:
        if not hasattr(self, "histogram_canvas"):
            return
        canvas = self.histogram_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 360)
        height = max(canvas.winfo_height(), 115)

        if self.histogram_mode == "today":
            self._draw_today_hourly_bars(canvas, width, height)
        else:
            self._draw_alltime_daily_bars(canvas, width, height)

    def _draw_today_hourly_bars(self, canvas, width: int, height: int) -> None:
        today_str = date.today().isoformat()
        day_map = (self.stats.get("hourly_data") or {}).get(today_str, {})
        values = [int(day_map.get(f"{h:02d}", day_map.get(str(h), 0))) for h in range(24)]
        max_val = max(values) if any(values) else 1

        pad_l, pad_r, pad_t, pad_b = 12, 12, 18, 22
        plot_w = max(40, width - pad_l - pad_r)
        plot_h = max(30, height - pad_t - pad_b)
        slot_w = plot_w / 24.0
        bar_w = max(4.0, slot_w * 0.68)

        if not any(values):
            canvas.create_text(
                width // 2,
                height // 2,
                text="No words dictated yet today",
                fill=self.colors["text_faint"],
                font=ui_font(10),
            )
            return

        now_hour = datetime.now().hour
        for hr, val in enumerate(values):
            x0 = pad_l + hr * slot_w + (slot_w - bar_w) / 2.0
            x1 = x0 + bar_w
            bar_h = (val / float(max_val)) * plot_h if val > 0 else 2.0
            y1 = pad_t + plot_h
            y0 = y1 - bar_h
            fill = (
                self.colors["teal"]
                if hr == now_hour and val > 0
                else (self.colors["batch"] if val > 0 else self.colors["border"])
            )
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
            if hr % 4 == 0:
                canvas.create_text(
                    x0 + bar_w / 2.0,
                    height - 10,
                    text=f"{hr:02d}h",
                    fill=self.colors["text_faint"],
                    font=mono_font(8),
                )

    def _draw_alltime_daily_bars(self, canvas, width: int, height: int) -> None:
        hourly_all = self.stats.get("hourly_data") or {}
        days_list = [(date.today() - timedelta(days=offset)).isoformat() for offset in range(13, -1, -1)]
        values = []
        for d_iso in days_list:
            hmap = hourly_all.get(d_iso, {})
            values.append(sum(int(v) for v in hmap.values()) if isinstance(hmap, dict) else 0)

        max_val = max(values) if any(values) else 1
        pad_l, pad_r, pad_t, pad_b = 12, 12, 18, 22
        plot_w = max(40, width - pad_l - pad_r)
        plot_h = max(30, height - pad_t - pad_b)
        slot_w = plot_w / float(len(days_list))
        bar_w = max(6.0, slot_w * 0.65)

        if not any(values):
            canvas.create_text(
                width // 2,
                height // 2,
                text="No 14-day history yet",
                fill=self.colors["text_faint"],
                font=ui_font(10),
            )
            return

        for idx, (d_iso, val) in enumerate(zip(days_list, values)):
            x0 = pad_l + idx * slot_w + (slot_w - bar_w) / 2.0
            x1 = x0 + bar_w
            bar_h = (val / float(max_val)) * plot_h if val > 0 else 2.0
            y1 = pad_t + plot_h
            y0 = y1 - bar_h
            fill = self.colors["teal"] if val > 0 else self.colors["border"]
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
            if idx in (0, 4, 9, 13):
                canvas.create_text(
                    x0 + bar_w / 2.0,
                    height - 10,
                    text=d_iso[5:],
                    fill=self.colors["text_faint"],
                    font=mono_font(8),
                )

    # ------------------------------------------------------------------
    # Speech Insights Accordion Section (10 Tabs + Visualizations)
    # ------------------------------------------------------------------

    def build_speech_insights_content(self, parent) -> None:
        import tkinter as tk
        import customtkinter as ctk

        # Range selector bar + Refresh button
        range_bar = tk.Frame(parent, bg=self.colors["accent_bg"])
        range_bar.pack(fill="x", pady=(2, 6))

        tk.Label(
            range_bar,
            text="Window:",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=ui_font(10),
        ).pack(side="left", padx=(0, 4))

        self._range_buttons: Dict[str, Any] = {}
        for label in INSIGHTS_RANGE_OPTIONS:
            is_sel = label == self.insights_range_label
            btn = ctk.CTkButton(
                range_bar,
                text=label,
                width=58,
                height=24,
                corner_radius=5,
                font=ui_font(10, "bold" if is_sel else None),
                fg_color=self.colors["teal_dim"] if is_sel else self.colors["bg"],
                hover_color=self.colors["teal"],
                text_color=self.colors["fg"] if is_sel else self.colors["text_muted"],
                command=lambda lbl=label: self.set_insights_range(lbl),
            )
            btn.pack(side="left", padx=2)
            self._range_buttons[label] = btn

        ctk.CTkButton(
            range_bar,
            text="Refresh",
            width=62,
            height=24,
            corner_radius=5,
            font=ui_font(10),
            fg_color=self.colors["border"],
            hover_color=self.colors["teal_dim"],
            text_color=self.colors["fg"],
            command=self.refresh_speech_insights,
        ).pack(side="right")

        # 2-row tab pill grid so all 10 tabs fit cleanly within 480px width
        tabs_grid = tk.Frame(parent, bg=self.colors["accent_bg"])
        tabs_grid.pack(fill="x", pady=(0, 6))
        for col in range(5):
            tabs_grid.grid_columnconfigure(col, weight=1)

        self._insight_tab_buttons: Dict[str, Any] = {}
        for idx, tab_name in enumerate(self.INSIGHT_TABS):
            r, c = divmod(idx, 5)
            active = tab_name == self.active_insight_tab
            btn = ctk.CTkButton(
                tabs_grid,
                text=tab_name,
                height=25,
                corner_radius=5,
                font=ui_font(10, "bold" if active else None),
                fg_color=self.colors["teal_dim"] if active else self.colors["bg"],
                hover_color=self.colors["teal"],
                text_color=self.colors["fg"] if active else self.colors["text_muted"],
                command=lambda t=tab_name: self.switch_insight_tab(t),
            )
            btn.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
            self._insight_tab_buttons[tab_name] = btn

        self.insights_body = tk.Frame(parent, bg=self.colors["accent_bg"])
        self.insights_body.pack(fill="x", pady=(2, 4))
        self.refresh_speech_insights()

    def set_insights_range(self, label: str) -> None:
        if label not in INSIGHTS_RANGE_OPTIONS:
            return
        self.insights_range_label = label
        self.config["insights_range"] = label
        self.save_config()
        for lbl, btn in getattr(self, "_range_buttons", {}).items():
            sel = lbl == label
            btn.configure(
                fg_color=self.colors["teal_dim"] if sel else self.colors["bg"],
                text_color=self.colors["fg"] if sel else self.colors["text_muted"],
                font=ui_font(10, "bold" if sel else None),
            )
        self.refresh_speech_insights()

    def switch_insight_tab(self, tab_name: str) -> None:
        if tab_name not in self.INSIGHT_TABS:
            return
        self.active_insight_tab = tab_name
        for name, btn in getattr(self, "_insight_tab_buttons", {}).items():
            sel = name == tab_name
            btn.configure(
                fg_color=self.colors["teal_dim"] if sel else self.colors["bg"],
                text_color=self.colors["fg"] if sel else self.colors["text_muted"],
                font=ui_font(10, "bold" if sel else None),
            )
        self._render_active_insight_tab()

    def refresh_speech_insights(self) -> None:
        all_entries = load_all_transcript_entries(paths.get_transcriptions_dir())
        days = INSIGHTS_RANGE_OPTIONS.get(self.insights_range_label, 30)
        window_entries = filter_entries_by_days(all_entries, days)
        self._cached_insights = compute_speech_insights(
            window_entries,
            all_entries_for_baseline=all_entries,
        )
        self._all_entries_cache = all_entries
        self._render_active_insight_tab()

    def _render_active_insight_tab(self) -> None:
        import tkinter as tk

        if not hasattr(self, "insights_body"):
            return
        for child in self.insights_body.winfo_children():
            child.destroy()

        data = self._cached_insights or {"empty": True}
        tab = self.active_insight_tab

        if tab == "AI Brief":
            self._render_ai_brief_tab(self.insights_body)
            return

        if tab == "Notes":
            self._render_notes_tab(self.insights_body, data)
            return

        if data.get("empty"):
            tk.Label(
                self.insights_body,
                text=(
                    f"No archived transcriptions found for '{self.insights_range_label}'.\n"
                    "Dictate a phrase using Live or Batch mode to populate Speech Insights."
                ),
                bg=self.colors["bg"],
                fg=self.colors["text_muted"],
                font=ui_font(10),
                justify="center",
                padx=12,
                pady=18,
            ).pack(fill="x")
            return

        renderers = {
            "Summary": self._render_summary_tab,
            "Trends": self._render_trends_tab,
            "Interventions": self._render_interventions_tab,
            "Pace": self._render_pace_tab,
            "Pause": self._render_pause_tab,
            "Clarity": self._render_clarity_tab,
            "Lexical": self._render_lexical_tab,
            "Activity": self._render_activity_tab,
        }
        fn = renderers.get(tab, self._render_summary_tab)
        fn(self.insights_body, data)

    # ------------------------------------------------------------------
    # Individual Tab Renderers
    # ------------------------------------------------------------------

    def _make_kpi_row(self, parent, items: List[Tuple[str, str, str]]) -> None:
        import tkinter as tk

        row = tk.Frame(parent, bg=self.colors["accent_bg"])
        row.pack(fill="x", pady=(0, 6))
        for idx, (title, val, sub) in enumerate(items):
            card = tk.Frame(
                row,
                bg=self.colors["bg"],
                highlightthickness=1,
                highlightbackground=self.colors["border"],
                padx=8,
                pady=6,
            )
            card.pack(side="left", expand=True, fill="both", padx=(0 if idx == 0 else 4, 0))
            tk.Label(card, text=title, bg=self.colors["bg"], fg=self.colors["text_muted"], font=ui_font(9)).pack(
                anchor="w"
            )
            tk.Label(card, text=val, bg=self.colors["bg"], fg=self.colors["teal"], font=ui_font(13, "bold")).pack(
                anchor="w"
            )
            if sub:
                tk.Label(
                    card, text=sub, bg=self.colors["bg"], fg=self.colors["text_faint"], font=ui_font(8)
                ).pack(anchor="w")

    def _render_summary_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        self._make_kpi_row(
            parent,
            [
                ("Words", f"{data['total_words']:,}", f"{data['entry_count']} utterances"),
                ("Time Saved", f"{data['time_saved_minutes']:.1f}m", f"vs {TYPING_WPM} WPM typing"),
                ("Clarity", f"{data['fillers_per_100']:.1f}%", "fillers / 100w"),
            ],
        )

        # Pentagon Speech Profile Radar Chart
        radar_canvas = tk.Canvas(
            parent,
            height=190,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        radar_canvas.pack(fill="x", pady=(2, 4))
        axes = data.get("radar_axes", [])
        radar_canvas.bind(
            "<Configure>",
            lambda _e, c=radar_canvas, ax=axes: self._draw_radar_chart(c, ax),
        )
        self.root.after(20, lambda c=radar_canvas, ax=axes: self._draw_radar_chart(c, ax))

    def _draw_radar_chart(self, canvas, axes: List[Tuple[str, float]]) -> None:
        if not axes or not canvas.winfo_exists():
            return
        canvas.delete("all")
        w = max(canvas.winfo_width(), 360)
        h = max(canvas.winfo_height(), 190)
        cx, cy = w / 2.0, h / 2.0 + 4.0
        radius = min(w, h) * 0.34
        n = len(axes)

        # Concentric reference webs
        for ring in (0.33, 0.66, 1.0):
            pts = []
            for i in range(n):
                ang = -math.pi / 2.0 + (2.0 * math.pi * i / n)
                pts.extend([cx + radius * ring * math.cos(ang), cy + radius * ring * math.sin(ang)])
            canvas.create_polygon(pts, outline=self.colors["border"], fill="", width=1)

        # Spokes + labels + data polygon
        poly_pts = []
        for i, (label, score) in enumerate(axes):
            ang = -math.pi / 2.0 + (2.0 * math.pi * i / n)
            ox = cx + radius * math.cos(ang)
            oy = cy + radius * math.sin(ang)
            canvas.create_line(cx, cy, ox, oy, fill=self.colors["border"])

            norm = max(0.08, min(1.0, float(score) / 100.0))
            px = cx + radius * norm * math.cos(ang)
            py = cy + radius * norm * math.sin(ang)
            poly_pts.extend([px, py])

            lx = cx + (radius + 26.0) * math.cos(ang)
            ly = cy + (radius + 16.0) * math.sin(ang)
            canvas.create_text(
                lx,
                ly,
                text=f"{label} ({int(round(score))})",
                fill=self.colors["text_muted"],
                font=ui_font(9),
            )

        if len(poly_pts) >= 6:
            canvas.create_polygon(
                poly_pts,
                outline=self.colors["teal"],
                fill=self.colors["teal_dim"],
                stipple="gray50",
                width=2,
            )

    def _render_ai_brief_tab(self, parent) -> None:
        import tkinter as tk
        import customtkinter as ctk

        top_bar = tk.Frame(parent, bg=self.colors["accent_bg"])
        top_bar.pack(fill="x", pady=(0, 6))

        self.btn_crunch_history = ctk.CTkButton(
            top_bar,
            text="Synthesize AI Brief" if not self._ai_crunch_running else "Synthesizing...",
            height=28,
            corner_radius=6,
            font=ui_font(11, "bold"),
            fg_color=self.colors["online"],
            hover_color="#9333ea",
            text_color=self.colors["fg"],
            state="disabled" if self._ai_crunch_running else "normal",
            command=self._start_ai_crunch_async,
        )
        self.btn_crunch_history.pack(side="left", padx=(0, 8))

        self.ai_auto_var = tk.BooleanVar(value=bool(self.config.get("ai_insights_auto", False)))
        ctk.CTkCheckBox(
            top_bar,
            text="Auto-refresh daily",
            variable=self.ai_auto_var,
            font=ui_font(10),
            checkbox_width=18,
            checkbox_height=18,
            fg_color=self.colors["teal"],
            command=self._on_toggle_ai_auto,
        ).pack(side="left")

        # Latest AI Daily Brief Card
        sorted_days = sorted(self._ai_insights_data.keys(), reverse=True)
        card = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=8,
        )
        card.pack(fill="x", pady=(0, 6))

        if not sorted_days:
            tk.Label(
                card,
                text=(
                    "AI Daily Brief uses your configured OpenAI API key (GPT-4o-mini) to summarize "
                    "themes, active projects, and tone across your recent transcriptions."
                ),
                bg=self.colors["bg"],
                fg=self.colors["text_muted"],
                font=ui_font(10),
                wraplength=410,
                justify="left",
            ).pack(anchor="w")
            return

        latest_day = sorted_days[0]
        latest = self._ai_insights_data.get(latest_day, {})
        mood = latest.get("mood", "Focused")
        brief = latest.get("daily_brief", "")
        themes = ", ".join(latest.get("themes") or []) or "—"

        tk.Label(
            card,
            text=f"{latest_day}  •  Tone: {mood}",
            bg=self.colors["bg"],
            fg=self.colors["teal"],
            font=ui_font(10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=brief or "No summary text recorded.",
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            font=ui_font(10),
            wraplength=410,
            justify="left",
        ).pack(anchor="w", pady=(3, 3))
        tk.Label(
            card,
            text=f"Themes: {themes}",
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
            wraplength=410,
            justify="left",
        ).pack(anchor="w")

        # Project Timeline & Mood Sparkline Controls (100% CTkOptionMenu — zero tk.OptionMenu!)
        ctrl_row = tk.Frame(parent, bg=self.colors["accent_bg"])
        ctrl_row.pack(fill="x", pady=(2, 4))

        tk.Label(
            ctrl_row,
            text="Timeline:",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
        ).pack(side="left")

        self.ai_timeline_dropdown = ctk.CTkOptionMenu(
            ctrl_row,
            values=["7 days", "14 days", "30 days"],
            width=92,
            height=24,
            font=ui_font(10),
            fg_color=self.colors["bg"],
            button_color=self.colors["border"],
            command=self._on_ai_timeline_range_change,
        )
        self.ai_timeline_dropdown.set(self._ai_timeline_days)
        self.ai_timeline_dropdown.pack(side="left", padx=(4, 10))

        tk.Label(
            ctrl_row,
            text="Mood:",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
        ).pack(side="left")

        self.ai_mood_dropdown = ctk.CTkOptionMenu(
            ctrl_row,
            values=["7 days", "14 days", "30 days"],
            width=92,
            height=24,
            font=ui_font(10),
            fg_color=self.colors["bg"],
            button_color=self.colors["border"],
            command=self._on_ai_mood_range_change,
        )
        self.ai_mood_dropdown.set(self._ai_mood_days)
        self.ai_mood_dropdown.pack(side="left", padx=(4, 0))

        # Gantt Project Swimlanes + Mood Sparkline Canvas
        spark_canvas = tk.Canvas(
            parent,
            height=110,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        spark_canvas.pack(fill="x", pady=(2, 2))
        self.root.after(20, lambda c=spark_canvas: self._draw_ai_mood_and_gantt(c))

    def _on_toggle_ai_auto(self) -> None:
        self.config["ai_insights_auto"] = bool(self.ai_auto_var.get())
        self.save_config()

    def _on_ai_timeline_range_change(self, val: str) -> None:
        self._ai_timeline_days = val
        self._render_active_insight_tab()

    def _on_ai_mood_range_change(self, val: str) -> None:
        self._ai_mood_days = val
        self._render_active_insight_tab()

    def _draw_ai_mood_and_gantt(self, canvas) -> None:
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        w = max(canvas.winfo_width(), 360)
        h = max(canvas.winfo_height(), 110)

        n_days = int(self._ai_mood_days.split()[0])
        day_keys = [(date.today() - timedelta(days=i)).isoformat() for i in range(n_days - 1, -1, -1)]
        pts = []
        for idx, dk in enumerate(day_keys):
            rec = self._ai_insights_data.get(dk)
            if isinstance(rec, dict) and "mood_score" in rec:
                score = max(1.0, min(5.0, float(rec.get("mood_score", 3.0))))
                x = 24.0 + (idx / float(max(1, n_days - 1))) * (w - 48.0)
                y = h - 22.0 - ((score - 1.0) / 4.0) * (h - 44.0)
                pts.append((x, y, dk[5:]))

        canvas.create_text(
            12,
            12,
            anchor="w",
            text=f"Mood Trajectory ({self._ai_mood_days})",
            fill=self.colors["text_muted"],
            font=ui_font(9, "bold"),
        )
        if len(pts) >= 2:
            flat = [coord for x, y, _ in pts for coord in (x, y)]
            canvas.create_line(flat, fill=self.colors["online"], width=2)
        for x, y, lbl in pts:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=self.colors["teal"], outline="")

    def _start_ai_crunch_async(self) -> None:
        api_key = str(self.config.get("openai_api_key", "") or "").strip()
        if not api_key:
            self.set_status("Set an OpenAI API key in Configuration -> Online to synthesize AI Briefs")
            return
        if self._ai_crunch_running:
            return
        self._ai_crunch_running = True
        self._render_active_insight_tab()

        def _worker() -> None:
            try:
                from openai import OpenAI

                entries = load_all_transcript_entries(paths.get_transcriptions_dir())
                recent = filter_entries_by_days(entries, 7)
                by_day: Dict[str, List[str]] = defaultdict(list)
                for e in recent:
                    by_day[e["date"]].append(e["text"])

                client = OpenAI(api_key=api_key)
                for day_iso, texts in sorted(by_day.items())[-5:]:
                    joined = "\n".join(texts)[:6000]
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Summarize the user's dictated notes for this day into JSON with keys: "
                                    "'daily_brief' (1-2 concise sentences), 'mood' (1-2 words), "
                                    "'mood_score' (number 1..5), 'themes' (list of up to 4 short tags), "
                                    "'projects' (list of up to 3 project names)."
                                ),
                            },
                            {"role": "user", "content": f"<transcript>\n{joined}\n</transcript>"},
                        ],
                        temperature=0.2,
                        max_tokens=350,
                    )
                    parsed = json.loads(resp.choices[0].message.content or "{}")
                    if isinstance(parsed, dict):
                        self._ai_insights_data[day_iso] = parsed
                self._save_ai_insights_disk()
            except Exception as exc:
                log.warning("AI Brief crunch error: %s", exc)
            finally:
                self._ai_crunch_running = False
                try:
                    self.root.after(0, self._render_active_insight_tab)
                except Exception:
                    pass

        threading.Thread(target=_worker, name="neurowhisper-ai-brief", daemon=True).start()

    def _render_trends_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        daily_words = data.get("daily_words") or {}
        sorted_days = sorted(daily_words.keys())[-14:]
        baseline_txt = (
            f"Personal 90d baseline active ({data['baseline_days']} days)"
            if data.get("has_personal_baseline")
            else f"Collecting personal baseline ({data.get('baseline_days', 0)}/{PERSONAL_BASELINE_MIN_DAYS} active days)"
        )
        self._make_kpi_row(
            parent,
            [
                ("Active Days", str(data["active_days"]), baseline_txt),
                ("Avg Words / Day", f"{int(round(data['total_words'] / max(1, data['active_days']))):,}", "active days"),
            ],
        )

        canvas = tk.Canvas(
            parent,
            height=130,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        canvas.pack(fill="x", pady=(2, 2))
        vals = [(d[5:], daily_words[d]) for d in sorted_days]
        self.root.after(20, lambda c=canvas, v=vals: self._draw_simple_bars(c, v, self.colors["teal"]))

    def _render_interventions_tab(self, parent, data: Dict[str, Any]) -> None:
        pb = data.get("baseline_metrics") or {}
        self._make_kpi_row(
            parent,
            [
                (
                    "Open Questions",
                    f"{data['open_q_per_100']:.1f}/100s",
                    _benchmark_status("intervention_open_question_per_100", data["open_q_per_100"], pb),
                ),
                (
                    "Reflection Ratio",
                    f"{data['reflection_ratio']:.2f}x",
                    _benchmark_status("intervention_reflection_ratio", data["reflection_ratio"], pb),
                ),
            ],
        )
        self._make_kpi_row(
            parent,
            [
                (
                    "Affirmations",
                    f"{data['affirmations_per_100']:.1f}/100s",
                    _benchmark_status("intervention_affirmation_per_100", data["affirmations_per_100"], pb),
                ),
                (
                    "Summaries",
                    f"{data['summaries_per_100']:.1f}/100s",
                    _benchmark_status("intervention_summary_per_100", data["summaries_per_100"], pb),
                ),
            ],
        )

    def _render_pace_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        pb = data.get("baseline_metrics") or {}
        self._make_kpi_row(
            parent,
            [
                (
                    "Avg Words / Entry",
                    f"{data['avg_words_entry']:.1f}",
                    _benchmark_status("pace_avg_words_entry", data["avg_words_entry"], pb),
                ),
                (
                    "Pace Stability",
                    f"{data['pace_stability']:.0f}/100",
                    _benchmark_status("pace_stability", data["pace_stability"], pb),
                ),
            ],
        )
        canvas = tk.Canvas(
            parent,
            height=115,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        canvas.pack(fill="x", pady=(2, 2))
        items = list((data.get("length_buckets") or {}).items())
        self.root.after(20, lambda c=canvas, v=items: self._draw_simple_bars(c, v, self.colors["batch"]))

    def _render_pause_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        pb = data.get("baseline_metrics") or {}
        self._make_kpi_row(
            parent,
            [
                (
                    "Avg Turn Gap",
                    f"{data['avg_pause_sec']:.1f}s",
                    _benchmark_status("pause_avg_seconds", data["avg_pause_sec"], pb),
                ),
                (
                    "Long Gaps (6s+)",
                    f"{data['long_pause_share'] * 100.0:.0f}%",
                    f"{data['pause_sample_count']} in-session transitions",
                ),
            ],
        )
        canvas = tk.Canvas(
            parent,
            height=115,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        canvas.pack(fill="x", pady=(2, 2))
        items = list((data.get("pause_buckets") or {}).items())
        self.root.after(20, lambda c=canvas, v=items: self._draw_simple_bars(c, v, self.colors["transcribing"]))

    def _render_clarity_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        pb = data.get("baseline_metrics") or {}
        self._make_kpi_row(
            parent,
            [
                (
                    "Fillers / 100w",
                    f"{data['fillers_per_100']:.2f}",
                    _benchmark_status("clarity_fillers_per_100", data["fillers_per_100"], pb),
                ),
                (
                    "Repetitions / 1k",
                    f"{data['reps_per_1k']:.1f}",
                    _benchmark_status("clarity_repetition_per_1k", data["reps_per_1k"], pb),
                ),
            ],
        )
        breakdown = list((data.get("filler_breakdown") or {}).items())
        if not breakdown:
            tk.Label(
                parent,
                text="Zero filler phrases detected in this window!",
                bg=self.colors["bg"],
                fg=self.colors["ready"],
                font=ui_font(10, "bold"),
                pady=12,
            ).pack(fill="x")
            return
        canvas = tk.Canvas(
            parent,
            height=115,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        canvas.pack(fill="x", pady=(2, 2))
        self.root.after(20, lambda c=canvas, v=breakdown: self._draw_simple_bars(c, v, self.colors["live"]))

    def _render_lexical_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        pb = data.get("baseline_metrics") or {}
        self._make_kpi_row(
            parent,
            [
                (
                    "Lexical Diversity",
                    f"{data['lexical_diversity']:.1f}%",
                    _benchmark_status("lexical_diversity_percent", data["lexical_diversity"], pb),
                ),
                (
                    "Complex Words (7+)",
                    f"{data['long_word_pct']:.1f}%",
                    _benchmark_status("lexical_long_word_percent", data["long_word_pct"], pb),
                ),
            ],
        )
        top_vocab = data.get("top_vocab") or []
        if top_vocab:
            box = tk.Frame(
                parent,
                bg=self.colors["bg"],
                highlightthickness=1,
                highlightbackground=self.colors["border"],
                padx=8,
                pady=6,
            )
            box.pack(fill="x")
            tk.Label(
                box,
                text="Top Content Terms:",
                bg=self.colors["bg"],
                fg=self.colors["text_muted"],
                font=ui_font(9, "bold"),
            ).pack(anchor="w")
            chips = "  •  ".join(f"{word} ({cnt})" for word, cnt in top_vocab[:10])
            tk.Label(
                box,
                text=chips,
                bg=self.colors["bg"],
                fg=self.colors["fg"],
                font=ui_font(9),
                wraplength=415,
                justify="left",
            ).pack(anchor="w", pady=(3, 0))

    def _render_activity_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk
        import customtkinter as ctk

        hdr = tk.Frame(parent, bg=self.colors["accent_bg"])
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(
            hdr,
            text="Contribution Heatmap Metric:",
            bg=self.colors["accent_bg"],
            fg=self.colors["text_muted"],
            font=ui_font(10),
        ).pack(side="left")

        self.activity_metric_dropdown = ctk.CTkOptionMenu(
            hdr,
            values=["Words", "Sessions", "Fillers"],
            width=105,
            height=24,
            font=ui_font(10),
            fg_color=self.colors["bg"],
            button_color=self.colors["border"],
            command=self._on_activity_metric_change,
        )
        self.activity_metric_dropdown.set(self._activity_metric)
        self.activity_metric_dropdown.pack(side="right")

        heatmap = tk.Canvas(
            parent,
            height=125,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        heatmap.pack(fill="x", pady=(2, 2))
        self.root.after(20, lambda c=heatmap, d=data: self._draw_github_heatmap(c, d))

    def _on_activity_metric_change(self, val: str) -> None:
        self._activity_metric = val
        self._render_active_insight_tab()

    def _draw_github_heatmap(self, canvas, data: Dict[str, Any]) -> None:
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        metric_key = {
            "Words": "daily_words",
            "Sessions": "daily_sessions",
            "Fillers": "daily_fillers",
        }.get(self._activity_metric, "daily_words")
        day_map = data.get(metric_key) or {}
        max_v = max(day_map.values()) if day_map else 1

        weeks = 16
        cell = 11
        gap = 3
        start_x = 32
        start_y = 18
        today_d = date.today()
        # Align end column to current week
        end_monday = today_d - timedelta(days=today_d.weekday())
        first_monday = end_monday - timedelta(weeks=weeks - 1)

        day_labels = ("Mon", "", "Wed", "", "Fri", "", "Sun")
        for r_idx, dl in enumerate(day_labels):
            if dl:
                canvas.create_text(
                    8,
                    start_y + r_idx * (cell + gap) + cell / 2,
                    anchor="w",
                    text=dl,
                    fill=self.colors["text_faint"],
                    font=mono_font(8),
                )

        palette = ("#161b22", "#134e4a", "#0f766e", "#14b8a6", "#2dd4bf")
        for wk in range(weeks):
            for dow in range(7):
                cur = first_monday + timedelta(weeks=wk, days=dow)
                if cur > today_d:
                    continue
                val = int(day_map.get(cur.isoformat(), 0))
                if val <= 0:
                    color = palette[0]
                else:
                    lvl = min(4, max(1, int(math.ceil((val / float(max(1, max_v))) * 4.0))))
                    color = palette[lvl]
                x0 = start_x + wk * (cell + gap)
                y0 = start_y + dow * (cell + gap)
                canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="")

    def _render_notes_tab(self, parent, data: Dict[str, Any]) -> None:
        import tkinter as tk

        lang_summary = (
            f"Detected language mix: {data.get('en_entries', 0)} EN / {data.get('de_entries', 0)} DE utterances."
            if not data.get("empty")
            else "Bilingual EN + DE filler & stopword analysis enabled automatically."
        )
        note_text = (
            f"{PEER_BENCHMARK_NOTE}\n\n"
            f"• Benchmark profile: {PEER_PROFILE['label']}.\n"
            f"• Personal Baseline: Once you accumulate {PERSONAL_BASELINE_MIN_DAYS}+ active dictation days "
            f"over {PERSONAL_BASELINE_DAYS} days, bands adapt to ±{int(PERSONAL_BASELINE_BAND_FRACTION * 100)}% "
            "of your personal rolling average.\n"
            f"• {lang_summary}"
        )
        card = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=10,
        )
        card.pack(fill="x")
        tk.Label(
            card,
            text=note_text,
            bg=self.colors["bg"],
            fg=self.colors["text_muted"],
            font=ui_font(9),
            wraplength=415,
            justify="left",
        ).pack(anchor="w")

    def _draw_simple_bars(self, canvas, items: List[Tuple[str, Any]], bar_color: str) -> None:
        if not canvas.winfo_exists() or not items:
            return
        canvas.delete("all")
        w = max(canvas.winfo_width(), 360)
        h = max(canvas.winfo_height(), 115)
        max_v = max(float(v) for _, v in items) if items else 1.0
        max_v = max(max_v, 1.0)

        pad_l, pad_r, pad_t, pad_b = 14, 14, 18, 22
        plot_w = max(40, w - pad_l - pad_r)
        plot_h = max(30, h - pad_t - pad_b)
        slot_w = plot_w / float(len(items))
        bar_w = min(44.0, max(8.0, slot_w * 0.62))

        for idx, (lbl, raw_val) in enumerate(items):
            val = float(raw_val)
            x0 = pad_l + idx * slot_w + (slot_w - bar_w) / 2.0
            x1 = x0 + bar_w
            bh = (val / max_v) * plot_h if val > 0 else 2.0
            y1 = pad_t + plot_h
            y0 = y1 - bh
            canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=bar_color if val > 0 else self.colors["border"],
                outline="",
            )
            if val > 0:
                canvas.create_text(
                    (x0 + x1) / 2.0,
                    max(9.0, y0 - 7.0),
                    text=f"{int(val)}" if val.is_integer() else f"{val:.1f}",
                    fill=self.colors["fg"],
                    font=mono_font(8),
                )
            canvas.create_text(
                (x0 + x1) / 2.0,
                h - 10,
                text=str(lbl)[:9],
                fill=self.colors["text_faint"],
                font=ui_font(8),
            )
