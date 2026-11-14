"""Comprehensive unit and GUI integration tests for Speech Insights 2.0.

Covers:
  * MATTR-50 length-invariance vs raw TTR sample-size bias + <50 Heap's-law projection
  * 16-bin pause histogram P90 interpolation & acoustic segment metadata extraction
  * Bilingual EN/DE lexical & timing proxies for brain fog, self-focus, rigidity,
    vitality vs strain, circadian late-night share, and Clarity Index scoring
  * Cross-language EN/DE offset alignment in 'All' mode vs 'EN' / 'DE' isolation
  * LexicalDiskIndex cold build, warm reload (0 rebuilt months), single-month
    incremental invalidation, Dropbox conflicted-copy exclusion, forward acoustic
    sidecar (.jsonl), and exact streaming parity across the <50 -> >=50 word boundary
  * Horizon slicing (Today, 7d, 30d, 90d, 6m, YTD, 12m, Prev 6m, Year YYYY, All)
    and Auto / Daily / Weekly / Monthly bucketing
  * Headless Xvfb WhisperApp integration: all 12 tabs, pill bars, breakdown toggles,
    crosshair HUD hover clamping, scroll clamping, Wide Longitudinal Trends Explorer,
    offline HTML report export (Section 9 SVGs), and Live/Batch acoustic sidecar capture.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
import tempfile
import time
import traceback
import types
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import lexical_insights  # noqa: E402
from test_stability_and_backends import (  # noqa: E402
    _get_or_load_whisper_gui_module,
    _install_safe_hardware_stubs,
    _tk_available,
)


FLUENT_EN_POOL = [
    "architecture", "pipeline", "calibration", "latency", "throughput", "synthesis",
    "validation", "hypothesis", "experiment", "instrument", "resilience", "diagnostic",
    "trajectory", "momentum", "granularity", "precision", "benchmark", "milestone",
    "clear", "focused", "energized", "productive", "curious", "grateful", "steady",
    "we", "our", "team", "built", "verified", "designed", "measured", "delivered",
    "system", "module", "interface", "network", "protocol", "database", "compiler",
    "schedule", "morning", "workshop", "analysis", "insights", "progress", "solution",
    "deployed", "tested", "documented", "optimized", "balanced", "coherent", "strong",
]

FLUENT_DE_POOL = [
    "architektur", "kalibrierung", "latenzzeit", "durchsatz", "validierung", "hypothese",
    "experiment", "werkzeug", "stabilität", "entwicklung", "präzision", "meilenstein",
    "klar", "fokussiert", "motiviert", "produktiv", "dankbar", "ausgeglichen", "stark",
    "wir", "unser", "gemeinsam", "gebaut", "geprüft", "gemessen", "geliefert", "geplant",
    "system", "schnittstelle", "datenbank", "protokoll", "analyse", "fortschritt", "lösung",
    "morgen", "konzept", "struktur", "ergebnis", "qualität", "übersicht", "verlässlich",
    "der", "die", "das", "und", "mit", "für", "auf", "aus", "bei", "nach", "über",
]


class TestMattrAndMathInvariants(unittest.TestCase):
    def test_mattr50_is_length_invariant_while_raw_ttr_collapses(self):
        rng = random.Random(42)
        vocab = [f"lexeme{i}" for i in range(65)]

        def make_stream(total_words: int) -> list[str]:
            out: list[str] = []
            while len(out) < total_words:
                block = list(vocab)
                rng.shuffle(block)
                out.extend(block)
            return out[:total_words]

        short_tokens = make_stream(200)
        long_tokens = make_stream(12000)

        raw_ttr_short = (len(set(short_tokens)) / len(short_tokens)) * 100.0
        raw_ttr_long = (len(set(long_tokens)) / len(long_tokens)) * 100.0
        # Raw TTR collapses with sample length (Heap's Law artifact)
        self.assertGreater(raw_ttr_short - raw_ttr_long, 25.0)

        s_sum, s_w = lexical_insights.compute_mattr(short_tokens, 50)
        l_sum, l_w = lexical_insights.compute_mattr(long_tokens, 50)
        mattr_short = (s_sum / s_w) * 100.0
        mattr_long = (l_sum / l_w) * 100.0
        # MATTR-50 stays invariant within ~2 percentage points across a 60x volume jump (>15x tighter than raw TTR)
        self.assertLess(abs(mattr_short - mattr_long), 2.5)

    def test_mattr_short_sample_floor_and_projection(self):
        # < 8 words -> ignored (0 weight) so 3-word commands don't skew diversity
        self.assertEqual(lexical_insights.compute_mattr(["hello", "world", "today"], 50), (0.0, 0.0))

        # 8 <= N < 50 -> bounded Heap's-law projection with reduced weight
        toks_20 = [f"word{i}" for i in range(20)]
        m_sum, m_w = lexical_insights.compute_mattr(toks_20, 50)
        self.assertAlmostEqual(m_w, (20.0 / 50.0) * 0.5, places=4)
        projected_pct = (m_sum / m_w) * 100.0
        self.assertGreaterEqual(projected_pct, 15.0)
        self.assertLessEqual(projected_pct, 96.0)

    def test_pause_p90_quantile_and_acoustic_segment_extraction(self):
        # Empty or all-zero histogram returns 0.0 (never 120.0s)
        self.assertEqual(lexical_insights.quantile_from_pause_hist([0] * 16, 5, 0.90), 0.0)

        fast_hist = [0] * len(lexical_insights.PAUSE_HIST_EDGES)
        fast_hist[1] = 18  # 0.5..1.0s
        fast_hist[2] = 2   # 1.0..1.5s
        p90_fast = lexical_insights.quantile_from_pause_hist(fast_hist, 20, 0.90)
        self.assertGreaterEqual(p90_fast, 0.8)
        self.assertLessEqual(p90_fast, 1.1)

        fog_hist = [0] * len(lexical_insights.PAUSE_HIST_EDGES)
        fog_hist[3] = 10   # 1.5..2.0s
        fog_hist[11] = 8   # 12..15s
        fog_hist[13] = 4   # 20..30s
        p90_fog = lexical_insights.quantile_from_pause_hist(fog_hist, 22, 0.90)
        self.assertGreaterEqual(p90_fog, 14.0)

        # Includes overlapping sub-segments (0.4..1.5 and 1.2..2.4 -> merged 0.4..2.4 = 2.0s)
        segs = [
            types.SimpleNamespace(start=0.4, end=1.5, text="Testing"),
            types.SimpleNamespace(start=1.2, end=2.4, text="internal"),
            types.SimpleNamespace(start=4.4, end=6.0, text="hesitation gaps now"),
        ]
        meta = lexical_insights.extract_acoustic_meta(8.0, segs)
        self.assertEqual(meta["audio_s"], 8.0)
        self.assertAlmostEqual(meta["speech_s"], 3.6, places=2)
        self.assertEqual(meta["pauses"], 1)
        self.assertEqual(meta["long_pauses"], 1)
        self.assertAlmostEqual(meta["max_pause_s"], 2.0, places=2)
        self.assertEqual(meta["words"], 5)

        merged = lexical_insights.merge_batch_acoustic_metas([meta, meta], total_words=10)
        self.assertEqual(merged["audio_s"], 16.0)
        self.assertAlmostEqual(merged["speech_s"], 7.2, places=2)
        self.assertEqual(merged["long_pauses"], 2)
        self.assertEqual(merged["words"], 10)


class TestLexicalProxiesAndCrossLanguageNormalization(unittest.TestCase):
    def _write_synthetic_archive(self, trans_dir: str) -> dt.date:
        os.makedirs(trans_dir, exist_ok=True)
        rng = random.Random(7)
        end_date = dt.date(2026, 4, 15)
        start_date = end_date - dt.timedelta(days=103)  # 90 baseline days + 14 fog days

        by_month: dict[str, list[str]] = {}
        cur = start_date
        while cur <= end_date:
            is_fog_phase = (end_date - cur).days < 14
            mkey = cur.strftime("%Y-%m")
            lines = by_month.setdefault(mkey, [])

            if not is_fog_phase:
                # Crisp, fluent bilingual baseline day (daytime 10:00, short 2s burst gaps)
                base_dt = dt.datetime.combine(cur, dt.time(10, 0, 0))
                for k in range(4):
                    pool = FLUENT_DE_POOL if (k % 2 == 1) else FLUENT_EN_POOL
                    words = [rng.choice(pool) for _ in range(35)]
                    ts = base_dt + dt.timedelta(seconds=k * 2)
                    lines.append(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(words)}.")
            else:
                # Simulated cognitive fog / high-strain episode (late night 23:30, 14s stalls)
                base_dt = dt.datetime.combine(cur, dt.time(23, 30, 0))
                fog_en = (
                    "I um I always feel completely exhausted and foggy today, maybe sort of "
                    "forgetting the the thing and stuff -- wait I mean whatever thing I needed "
                    "perhaps for my project, uh I just feel tired and drained."
                )
                fog_de = (
                    "Ich ähm ich bin immer völlig erschöpft und müde heute, vielleicht irgendwie "
                    "fehlt mir das das Ding und Zeug -- warte ich meine halt dieses Dingsbums "
                    "für mein Projekt, äh ich bin komplett ausgelaugt."
                )
                for k, txt in enumerate((fog_en, fog_de, fog_en, fog_de)):
                    ts = base_dt + dt.timedelta(seconds=k * 14)
                    lines.append(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {txt}")

            cur += dt.timedelta(days=1)

        for mkey, lines in by_month.items():
            with open(os.path.join(trans_dir, f"{mkey}.TESTPC.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

        return end_date

    def test_brain_fog_and_mental_health_proxies_shift_and_normalize_en_de(self):
        with tempfile.TemporaryDirectory(prefix="nw_lex_proxies_") as tmp:
            trans_dir = os.path.join(tmp, "transcriptions")
            end_date = self._write_synthetic_archive(trans_dir)
            idx = lexical_insights.LexicalDiskIndex(
                transcriptions_dir=trans_dir,
                cache_path=os.path.join(tmp, "lexical_cache.TESTPC.json"),
                hostname="TESTPC",
            )
            idx.refresh()

            # Compare clean baseline slice vs recent 7-day fog slice
            clean_start = (end_date - dt.timedelta(days=80)).isoformat()
            clean_end = (end_date - dt.timedelta(days=20)).isoformat()
            sl_clean = lexical_insights.query_slice(
                idx, range_spec=f"{clean_start}..{clean_end}", bucket_mode="Daily", lang_mode="All", today=end_date
            )
            sl_fog = lexical_insights.query_slice(
                idx, range_spec="7 days", bucket_mode="Daily", lang_mode="All", today=end_date
            )

            c_sum = sl_clean["summary"]
            f_sum = sl_fog["summary"]

            self.assertGreater(f_sum["vague_per_100"], c_sum["vague_per_100"] + 4.0)
            self.assertGreater(f_sum["fillers_per_100"], c_sum["fillers_per_100"] + 3.0)
            self.assertGreater(f_sum["disfluency_per_1k"], c_sum["disfluency_per_1k"] + 20.0)
            self.assertGreater(f_sum["pause_long_pct"], 90.0)
            self.assertLess(c_sum["pause_long_pct"], 5.0)
            self.assertGreater(f_sum["pause_p90_s"], 10.0)
            self.assertLess(f_sum["mattr_diversity"], c_sum["mattr_diversity"] - 5.0)
            self.assertGreater(f_sum["self_focus_per_100"], c_sum["self_focus_per_100"] + 5.0)
            self.assertGreater(f_sum["absolutist_per_100"], c_sum["absolutist_per_100"] + 1.5)
            self.assertGreater(f_sum["hedges_per_100"], c_sum["hedges_per_100"] + 2.0)
            # vitality_balance is centered at 50 (>50 = vitality, <50 = strain/fatigue)
            self.assertLess(f_sum["vitality_balance"], 35.0)
            self.assertGreater(c_sum["vitality_balance"], 55.0)
            self.assertEqual(f_sum["late_night_pct"], 100.0)
            self.assertEqual(c_sum["late_night_pct"], 0.0)

            # Composite Clarity Index drops sharply during the simulated fog week
            self.assertGreater(c_sum["clarity_index"], 48.0)
            self.assertLess(f_sum["clarity_index"], 35.0)
            self.assertTrue(sl_fog["callout"])

            # Language isolation modes ('EN' vs 'DE') strictly partition words & de_share_pct
            sl_en = lexical_insights.query_slice(
                idx, range_spec="30 days", bucket_mode="Weekly", lang_mode="EN", today=end_date
            )
            sl_de = lexical_insights.query_slice(
                idx, range_spec="30 days", bucket_mode="Weekly", lang_mode="DE", today=end_date
            )
            self.assertEqual(sl_en["summary"]["de_share_pct"], 0.0)
            self.assertEqual(sl_de["summary"]["de_share_pct"], 100.0)
            self.assertGreater(sl_en["summary"]["words"], 500)
            self.assertGreater(sl_de["summary"]["words"], 500)


class TestLexicalDiskIndexAndSlicing(unittest.TestCase):
    def test_incremental_disk_cache_watermarks_conflict_skip_and_live_crossover(self):
        with tempfile.TemporaryDirectory(prefix="nw_lex_disk_") as tmp:
            trans_dir = os.path.join(tmp, "transcriptions")
            os.makedirs(trans_dir, exist_ok=True)
            cache_path = os.path.join(tmp, "lexical_cache.HOSTA.json")

            months = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]
            for m in months:
                p = os.path.join(trans_dir, f"{m}.HOSTA.txt")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(
                        f"[{m}-10 09:00:00] {' '.join(FLUENT_EN_POOL[:30])}.\n"
                        f"[{m}-10 09:00:03] {' '.join(FLUENT_DE_POOL[:30])}.\n"
                    )

            # Peer machine file in 2026-03 (must be merged) + Dropbox conflicted copy (must be skipped)
            with open(os.path.join(trans_dir, "2026-03.HOSTB.txt"), "w", encoding="utf-8") as fh:
                fh.write("[2026-03-11 11:00:00] Peer machine valid transcript entry with ten distinct clear words.\n")
            with open(
                os.path.join(trans_dir, "2026-03.HOSTA (Laptop's conflicted copy 2026-03-11).txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write("[2026-03-11 11:00:00] Conflicted duplicate words that must never be counted.\n")

            idx1 = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx1.refresh()
            self.assertEqual(idx1.rebuilt_months, 6)
            self.assertEqual(idx1.scanned_files, 7)  # 6 HOSTA + 1 HOSTB (conflicted copy excluded)
            self.assertTrue(os.path.isfile(cache_path))
            self.assertEqual(idx1.daily["2026-03-11"]["entries"], 1)

            # Warm reload: 0 months rebuilt
            t0 = time.perf_counter()
            idx2 = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx2.refresh()
            warm_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(idx2.rebuilt_months, 0)
            self.assertLess(warm_ms, 50.0)

            # Modify only 1 month -> exactly 1 month rebuilt
            with open(os.path.join(trans_dir, "2026-03.HOSTA.txt"), "a", encoding="utf-8") as fh:
                fh.write("[2026-03-15 14:00:00] Appended single new entry in March to verify incremental rebuild.\n")
            idx3 = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx3.refresh()
            self.assertEqual(idx3.rebuilt_months, 1)

            # Test streaming record_live_entry across <50 -> >=50 word boundary + acoustic sidecar
            # Seed 2026-04-01 with 25 EN words (<50 Heap's-law regime)
            apr_path = os.path.join(trans_dir, "2026-04.HOSTA.txt")
            ts1 = dt.datetime(2026, 4, 1, 10, 0, 0)
            text1 = " ".join(FLUENT_EN_POOL[:25])
            with open(apr_path, "w", encoding="utf-8") as fh:
                fh.write(f"[{ts1.strftime('%Y-%m-%d %H:%M:%S')}] {text1}\n")
            lexical_insights.append_acoustic_meta(
                trans_dir, "HOSTA", ts1, {"audio_s": 12.0, "speech_s": 10.0, "pauses": 1, "long_pauses": 0, "words": 25}
            )

            idx_live = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx_live.refresh()
            self.assertEqual(idx_live.rebuilt_months, 1)

            # Stream two more entries on 2026-04-01 crossing >50 total EN words (25 + 20 + 20 = 65)
            for step, offset_s, chunk_words in (
                (1, 3, FLUENT_EN_POOL[10:30]),
                (2, 11, FLUENT_EN_POOL[25:45]),
            ):
                ts_step = ts1 + dt.timedelta(seconds=offset_s)
                txt_step = " ".join(chunk_words)
                ac_step = {"audio_s": 10.0, "speech_s": 7.5, "pauses": 2, "long_pauses": 1, "words": len(chunk_words)}
                # Mirror WhisperApp.save_transcription order: record_live_entry -> disk write -> sync_host_watermarks
                idx_live.record_live_entry(ts_step, txt_step, ac_step)
                with open(apr_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{ts_step.strftime('%Y-%m-%d %H:%M:%S')}] {txt_step}\n")
                lexical_insights.append_acoustic_meta(trans_dir, "HOSTA", ts_step, ac_step)
                idx_live.sync_host_watermarks(ts_step)
            idx_live.save()

            # Warm reload must rebuild 0 months because watermarks were synced
            idx_warm_after = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx_warm_after.refresh()
            self.assertEqual(idx_warm_after.rebuilt_months, 0)

            # Cold rebuild from raw files must match the streamed day record identically!
            idx_cold_verify = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx_cold_verify.refresh(force_rebuild=True)
            live_day = idx_warm_after.daily["2026-04-01"]
            cold_day = idx_cold_verify.daily["2026-04-01"]
            for key in ("entries", "words", "gap_count", "gap_long", "ac_audio_s", "ac_speech_s", "ac_words"):
                self.assertEqual(live_day[key], cold_day[key], f"Mismatch on {key}")
            self.assertAlmostEqual(live_day["en"]["mattr_sum"], cold_day["en"]["mattr_sum"], places=3)
            self.assertAlmostEqual(live_day["en"]["mattr_w"], cold_day["en"]["mattr_w"], places=3)

            # Verify slicing horizons & granularities
            today_ref = dt.date(2026, 4, 2)
            sl_ytd = lexical_insights.query_slice(idx_cold_verify, "This year", "Monthly", "All", today=today_ref)
            self.assertEqual(sl_ytd["start_date"], "2026-01-01")
            self.assertEqual(sl_ytd["bucket_mode"], "monthly")
            self.assertEqual(len(sl_ytd["points"]), 4)  # Jan, Feb, Mar, Apr 2026

            sl_y25 = lexical_insights.query_slice(idx_cold_verify, "Year 2025", "Monthly", "All", today=today_ref)
            self.assertEqual(sl_y25["start_date"], "2025-01-01")
            self.assertEqual(sl_y25["end_date"], "2025-12-31")
            self.assertEqual(sl_y25["active_days_in_slice"], 3)  # Oct, Nov, Dec 2025

            sl_6m = lexical_insights.query_slice(idx_cold_verify, "6 months", "Auto", "All", today=today_ref)
            self.assertEqual(sl_6m["bucket_mode"], "weekly")
            self.assertGreater(sl_6m["summary"]["acoustic_wpm"], 100.0)
            self.assertGreater(sl_6m["summary"]["acoustic_silence_pct"], 15.0)

            opts = lexical_insights.available_period_options(idx_cold_verify.daily, today=today_ref)
            self.assertIn("6 months", opts)
            self.assertIn("This year", opts)
            self.assertIn("Prev 6 months", opts)
            self.assertIn("Year 2025", opts)
            self.assertIn("Year 2026", opts)

    def test_midnight_month_rollover_and_partial_acoustic_meta(self):
        with tempfile.TemporaryDirectory(prefix="nw_lex_rollover_") as tmp:
            trans_dir = os.path.join(tmp, "transcriptions")
            os.makedirs(trans_dir, exist_ok=True)
            cache_path = os.path.join(tmp, "lexical_cache.HOSTA.json")

            idx_stream = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx_stream.refresh()

            # Stream across:
            # 1) intra-month midnight rollover: 2026-03-20 23:59:50 -> 2026-03-21 00:00:04 (14s gap)
            # 2) month-boundary rollover: 2026-03-31 23:59:52 -> 2026-04-01 00:00:05 (13s gap)
            events = [
                (dt.datetime(2026, 3, 20, 23, 59, 50), " ".join(FLUENT_EN_POOL[:15]), {"audio_s": 6.0, "words": 15}),
                (dt.datetime(2026, 3, 21, 0, 0, 4), " ".join(FLUENT_EN_POOL[15:30]), {"audio_s": 5.0, "words": 15}),
                (dt.datetime(2026, 3, 31, 23, 59, 52), " ".join(FLUENT_EN_POOL[:15]), None),
                (dt.datetime(2026, 4, 1, 0, 0, 5), " ".join(FLUENT_EN_POOL[10:25]), None),
            ]
            for ts_ev, txt_ev, ac_ev in events:
                idx_stream.record_live_entry(ts_ev, txt_ev, ac_ev)
                m_path = os.path.join(trans_dir, f"{ts_ev.strftime('%Y-%m')}.HOSTA.txt")
                with open(m_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{ts_ev.strftime('%Y-%m-%d %H:%M:%S')}] {txt_ev}\n")
                if ac_ev is not None:
                    lexical_insights.append_acoustic_meta(trans_dir, "HOSTA", ts_ev, ac_ev)
                idx_stream.sync_host_watermarks(ts_ev)

            # Inject malformed JSONL lines + non-dict JSON values into March meta file
            meta_mar = os.path.join(trans_dir, "meta", "2026-03.HOSTA.jsonl")
            with open(meta_mar, "a", encoding="utf-8") as fh:
                fh.write("{corrupt json line\n[1, 2, 3]\n\"just_a_string\"\n{\"d\": \"not-a-date\", \"speech_s\": 3.0}\n")

            idx_cold = lexical_insights.LexicalDiskIndex(trans_dir, cache_path, hostname="HOSTA")
            idx_cold.refresh(force_rebuild=True)

            # Intra-month midnight gap is preserved identically in both live and cold rebuilds
            self.assertEqual(idx_stream.daily["2026-03-21"]["gap_count"], 1)
            self.assertEqual(idx_cold.daily["2026-03-21"]["gap_count"], 1)
            self.assertEqual(idx_stream.daily["2026-03-21"]["gap_long"], 1)
            self.assertEqual(idx_cold.daily["2026-03-21"]["gap_long"], 1)

            # Month-boundary midnight gap stays isolated per month file in both live and cold rebuilds
            self.assertEqual(idx_stream.daily["2026-04-01"]["gap_count"], 0)
            self.assertEqual(idx_cold.daily["2026-04-01"]["gap_count"], 0)

            # Online-mode acoustic records without 'speech_s' must NOT fabricate 0% VAD silence
            m_21 = lexical_insights.compute_record_metrics(idx_cold.daily["2026-03-21"])
            self.assertIsNone(m_21["acoustic_silence_pct"])
            self.assertIsNone(m_21["acoustic_wpm"])



@unittest.skipUnless(_tk_available(), "Tk display unavailable (run under xvfb-run on headless Linux)")
class TestSpeechInsights2GuiIntegration(unittest.TestCase):
    def setUp(self):
        import tkinter as tk

        self.wg = _get_or_load_whisper_gui_module()
        _install_safe_hardware_stubs()
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="nw_si2_gui_", ignore_cleanup_errors=True)
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)
        self.orig_app_dir = self.wg.app_dir
        self.wg.app_dir = self.tmp_dir.name

        trans_dir = os.path.join(self.tmp_dir.name, "transcriptions")
        os.makedirs(trans_dir, exist_ok=True)
        self._seed_gui_archive(trans_dir)

        self.orig_kb = self.wg.keyboard
        self.stub_kb = sys.modules["keyboard"]
        self.wg.keyboard = self.stub_kb
        self.stub_kb.typed_texts = []
        self.stub_kb.sent_keys = []
        self.stub_kb.raise_on_send = False

        self._load_patch = mock.patch.object(self.wg.WhisperApp, "load_model", lambda self_app: None)
        self._sound_patch = mock.patch.object(self.wg.WhisperApp, "play_feedback_sound", lambda *a, **k: None)
        self._load_patch.start()
        self._sound_patch.start()

        self.tk = tk
        self.root = tk.Tk()
        self.root.geometry("480x820+40+40")
        self.app = self.wg.WhisperApp(self.root)
        # Attach callback exception tracker AFTER WhisperApp.__init__ sets its default hook
        self.tk_errors: list[str] = []
        self.root.report_callback_exception = (
            lambda exc, val, tb: self.tk_errors.append("".join(traceback.format_exception(exc, val, tb)))
        )

        class _TimedFakeBackend:
            def __init__(self):
                self.is_model_loaded = True
                self.calls = 0

            def transcribe_audio(self, audio, sample_rate=16000, **kwargs):
                self.calls += 1
                s1 = types.SimpleNamespace(
                    text=f"Clear spoken sentence number {self.calls} with steady cadence",
                    start=0.2,
                    end=1.8,
                )
                s2 = types.SimpleNamespace(
                    text="and measured lexical precision today.",
                    start=2.6,
                    end=3.9,
                )
                return [s1, s2]

            def unload_model(self):
                self.is_model_loaded = False

        self.fake_backend = _TimedFakeBackend()
        self.app.backend = self.fake_backend
        self.app.model = self.fake_backend
        self.app.current_backend_name = "faster-whisper"
        self.app.current_device = "cpu"
        self.app._model_loading = False
        self.app._model_load_failed = False

    def tearDown(self):
        try:
            self.app.on_close()
        except Exception:
            pass
        self._sound_patch.stop()
        self._load_patch.stop()
        self.wg.keyboard = self.orig_kb
        self.wg.app_dir = self.orig_app_dir
        os.chdir(self.orig_cwd)
        self.tmp_dir.cleanup()
        self.assertEqual(self.tk_errors, [], f"Tk callback errors:\n{''.join(self.tk_errors)}")

    def _pump_tk(self, duration_s: float = 0.12):
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)

    def _emit_canvas_motion(self, canvas, x: int, y: int):
        try:
            canvas.event_generate("<Motion>", x=x, y=y)
        except Exception:
            pass
        self._pump_tk(0.02)
        fn = getattr(canvas, "_nw_ts_motion_fn", None)
        if fn is not None and not canvas.find_withtag("ts_hover"):
            fn(types.SimpleNamespace(x=x, y=y))
            self._pump_tk(0.01)

    def _emit_canvas_leave(self, canvas):
        try:
            canvas.event_generate("<Leave>")
        except Exception:
            pass
        self._pump_tk(0.02)
        fn = getattr(canvas, "_nw_ts_leave_fn", None)
        if fn is not None and canvas.find_withtag("ts_hover"):
            fn(types.SimpleNamespace())
            self._pump_tk(0.01)

    @staticmethod
    def _seed_gui_archive(trans_dir: str):
        today = dt.date.today()
        # Seed entries in previous calendar year + trailing 120 days up to today
        prev_year = today.year - 1
        with open(os.path.join(trans_dir, f"{prev_year}-11.SEED.txt"), "w", encoding="utf-8") as fh:
            for day in (5, 12, 19, 26):
                fh.write(
                    f"[{prev_year}-11-{day:02d} 10:15:00] {' '.join(FLUENT_EN_POOL[:32])}.\n"
                    f"[{prev_year}-11-{day:02d} 10:15:03] {' '.join(FLUENT_DE_POOL[:32])}.\n"
                )

        by_month: dict[str, list[str]] = {}
        for back in range(95, -1, -3):
            d = today - dt.timedelta(days=back)
            mkey = d.strftime("%Y-%m")
            lines = by_month.setdefault(mkey, [])
            t_base = dt.datetime.combine(d, dt.time(11 if back > 9 else 23, 10, 0))
            en_txt = " ".join(FLUENT_EN_POOL[:30])
            de_txt = " ".join(FLUENT_DE_POOL[:30])
            gap_s = 2 if back > 9 else 9
            lines.append(f"[{t_base.strftime('%Y-%m-%d %H:%M:%S')}] {en_txt} um sort of thing.")
            t2 = t_base + dt.timedelta(seconds=gap_s)
            lines.append(f"[{t2.strftime('%Y-%m-%d %H:%M:%S')}] {de_txt} ähm irgendwie Ding.")

        for mkey, lines in by_month.items():
            with open(os.path.join(trans_dir, f"{mkey}.SEED.txt"), "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

    def test_twelve_tabs_slicing_crosshair_explorer_report_and_acoustic_sidecar(self):
        self._pump_tk(0.10)

        # 1. Expand Speech Insights if collapsed
        if self.app.insights_collapsed.get():
            self.app.toggle_insights()
        self._pump_tk(0.15)
        self.assertEqual(len(self.app.insight_tab_frames), 12)

        # 2. Cycle through all 12 tabs and verify lazy construction + rendering
        for tab_idx in range(12):
            self.app._select_insight_tab(tab_idx)
            self._pump_tk(0.04)
            self.assertIn(tab_idx, self.app._insight_tabs_built)

        self.assertGreater(len(self.app.fogwatch_canvas.find_all()), 5)
        self.assertGreater(len(self.app.mind_canvas.find_all()), 5)
        self.assertGreater(len(self.app.speaker_profile_canvas.find_all()), 5)

        # 3. Toggle metric pills & breakdown bars across tabs (including hover/resize isolation)
        self.app._select_insight_tab(0)  # FogWatch
        self.app.fogwatch_pill_buttons["vague_per_100"].invoke()
        self._pump_tk(0.03)
        self.assertGreater(len(self.app.fogwatch_canvas.find_all()), 5)
        self.app.fogwatch_pill_buttons["clarity_index"].invoke()
        self._pump_tk(0.03)

        self.app._select_insight_tab(2)  # Pauses: line vs breakdown
        self._emit_canvas_motion(self.app.pause_canvas, 80, 40)
        self.assertGreater(len(self.app.pause_canvas.find_withtag("ts_hover")), 0)
        # Switch to Breakdown bars -> stale hover crosshair & _nw_ts_args must clear
        self.app.pause_pill_buttons["breakdown"].invoke()
        self._pump_tk(0.03)
        self.assertGreater(len(self.app.pause_canvas.find_all()), 3)
        self.assertIsNone(getattr(self.app.pause_canvas, "_nw_ts_args", None))
        self.assertIsNone(getattr(self.app.pause_canvas, "_nw_ts_motion_fn", None))
        self._emit_canvas_motion(self.app.pause_canvas, 80, 40)
        self.assertEqual(len(self.app.pause_canvas.find_withtag("ts_hover")), 0)
        self.app.pause_pill_buttons["pause_p90_s"].invoke()
        self._pump_tk(0.03)
        self.assertGreater(len(self.app.pause_canvas.find_all()), 5)

        self.app._select_insight_tab(3)  # Clarity: breakdown vs repairs
        self.app.clarity_pill_buttons["breakdown"].invoke()
        self._pump_tk(0.03)
        self.app.clarity_pill_buttons["disfluency_per_1k"].invoke()
        self._pump_tk(0.03)

        self.app._select_insight_tab(4)  # Lexical: breakdown vs MATTR-50
        self.app.lexical_pill_buttons["breakdown"].invoke()
        self._pump_tk(0.03)
        self.app.lexical_pill_buttons["mattr_diversity"].invoke()
        self._pump_tk(0.03)

        self.app._select_insight_tab(5)  # Mind
        for m_key in ("vitality_balance", "self_focus_per_100", "absolutist_per_100", "hedges_per_100", "late_night_pct"):
            self.app.mind_pill_buttons[m_key].invoke()
            self._pump_tk(0.02)
            self.assertGreater(len(self.app.mind_canvas.find_all()), 5)

        self.app._select_insight_tab(9)  # Activity heatmap modes
        for mode_lbl in ("Clarity Index", "Lexical Diversity", "Long Pauses %", "Vitality", "Words (Volume)"):
            self.app.activity_metric_var.set(mode_lbl)
            self.app.refresh_insights_tabs()
            self._pump_tk(0.02)
            self.assertGreater(len(self.app.activity_canvas.find_all()), 5)

        # 4. Test Horizon Slicing, Bucket Granularity & Language Stratification
        prev_year_label = f"Year {dt.date.today().year - 1}"
        for horizon in ("6 months", "This year", prev_year_label, "All", "90 days"):
            self.app._select_insights_range(horizon)
            self._pump_tk(0.04)
            self.assertEqual(self.app.insights_range_var.get(), horizon)

        self.app.insights_bucket_var.set("Weekly")
        self.app.insights_lang_var.set("DE")
        self.app._on_insights_filter_change()
        self._pump_tk(0.04)
        self.assertEqual(self.app._current_lexical_slice["summary"]["de_share_pct"], 100.0)

        self.app.insights_bucket_var.set("Auto")
        self.app.insights_lang_var.set("All")
        self.app._on_insights_filter_change()
        self._pump_tk(0.04)

        # 5. Exercise interactive crosshair HUD tooltip & edge clamping on FogWatch canvas
        self.app._select_insight_tab(0)
        self._pump_tk(0.05)
        cw = max(120, self.app.fogwatch_canvas.winfo_width())
        ch = max(80, self.app.fogwatch_canvas.winfo_height())
        for hx, hy in ((60, 40), (cw - 4, 8), (cw - 2, ch - 4)):
            self._emit_canvas_motion(self.app.fogwatch_canvas, hx, hy)
            hover_items = self.app.fogwatch_canvas.find_withtag("ts_hover")
            self.assertGreater(len(hover_items), 0)
            bbox = self.app.fogwatch_canvas.bbox("ts_hover")
            self.assertIsNotNone(bbox)
            self.assertGreaterEqual(bbox[0], 0)
            self.assertLessEqual(bbox[2], cw + 2)
        self._emit_canvas_leave(self.app.fogwatch_canvas)
        self.assertEqual(len(self.app.fogwatch_canvas.find_withtag("ts_hover")), 0)

        # 6. Verify viewport scroll clamping holds with Speech Insights expanded
        for _ in range(10):
            self.app._on_mousewheel(types.SimpleNamespace(num=4, delta=120, widget=self.root), force_canvas=True)
        self._pump_tk(0.04)
        self.assertAlmostEqual(self.app._scroll_canvas.canvasy(0), 0.0, places=1)

        # 7. Open Wide Longitudinal Trends Explorer + dual-axis overlay + HTML report export
        self.app.open_trends_explorer()
        self._pump_tk(0.15)
        self.assertTrue(self.app._explorer_is_open())
        self.assertGreater(len(self.app._explorer_main_canvas.find_all()), 10)
        self.assertGreater(len(self.app._explorer_vol_canvas.find_all()), 3)

        self.app._select_explorer_primary_metric("mattr_diversity")
        self.app._explorer_secondary_var.set("None")
        self.app._refresh_trends_explorer()
        self._pump_tk(0.05)
        self.assertGreater(len(self.app._explorer_main_canvas.find_all()), 8)

        report_out = os.path.join(self.tmp_dir.name, "report_out")
        report_html_path = self.app._generate_html_report_sync(out_dir=report_out)
        self.assertTrue(os.path.isfile(report_html_path))
        with open(report_html_path, "r", encoding="utf-8") as fh:
            html = fh.read()
        self.assertIn("Longitudinal Lexical", html)
        self.assertIn("Cognitive Clarity Index", html)
        self.assertIn("<svg", html)

        # 8. Live chunk inference -> verify forward acoustic metadata sidecar & WPM/silence metrics
        chunk_5s = [np.ones((1600, 1), dtype=np.float32) * 0.15 for _ in range(50)]  # 5.0s audio
        self.app.process_live_chunk(chunk_5s)
        self.app._update_gui_loop_body()
        self._pump_tk(0.05)

        meta_dir = os.path.join(self.tmp_dir.name, "transcriptions", "meta")
        self.assertTrue(os.path.isdir(meta_dir))
        meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".jsonl")]
        self.assertEqual(len(meta_files), 1)
        with open(os.path.join(meta_dir, meta_files[0]), "r", encoding="utf-8") as fh:
            meta_rec = json.loads(fh.readline())
        self.assertAlmostEqual(meta_rec["audio_s"], 5.0, places=1)
        self.assertAlmostEqual(meta_rec["speech_s"], 2.9, places=1)
        self.assertGreater(meta_rec["words"], 8)

        # Refresh insights for Today and verify BOTH summary metrics AND timeseries charts render!
        self.app._select_insights_range("Today")
        self._pump_tk(0.05)
        today_sum = self.app._current_lexical_slice["summary"]
        self.assertIsNotNone(today_sum["acoustic_wpm"])
        self.assertGreater(today_sum["acoustic_wpm"], 100.0)
        self.assertIsNotNone(today_sum["acoustic_silence_pct"])
        self.assertGreater(today_sum["acoustic_silence_pct"], 30.0)
        self.assertEqual(self.app._current_lexical_slice["bucket_mode"], "daily")
        self.assertGreater(len(self.app.fogwatch_canvas.find_all()), 5)
        self.assertGreater(len(self.app._explorer_main_canvas.find_all()), 5)

        # Flush lexical cache via on_close and verify per-machine cache file exists
        self.app.on_close()
        cache_file = os.path.join(
            self.tmp_dir.name, f"lexical_cache.{self.wg.WhisperApp._hostname()}.json"
        )
        self.assertTrue(os.path.isfile(cache_file))


if __name__ == "__main__":
    unittest.main()
