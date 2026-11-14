"""Build ``analysis/out/series.json`` from labels, clusters and transcripts.

Fully local: no network, no API key. Combines
``out/labels/*.json`` (mood, topics, activities), ``out/clusters.json`` +
``out/categories.json`` (topic phrase -> category, optional) and the transcript
archive (words, entries, timing, language) into one series file that
``report.py`` renders.

Example::

    python analysis/build_series.py
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import statistics as st
import sys

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

import common


def mean(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), nd) if xs else None


def longest_streak(days):
    """Longest run of consecutive calendar days with any dictation."""
    best = run = 0
    prev = None
    for day in sorted(days):
        d = dt.date.fromisoformat(day)
        run = run + 1 if prev is not None and (d - prev).days == 1 else 1
        best = max(best, run)
        prev = d
    return best


def build(
    transcripts_dir=None,
    labels_dir=None,
    cache_path=None,
    out_dir=None,
    ai_insights_path=None,
    trim_outliers=False,
):
    entries = common.load_entries(transcripts_dir)
    if not entries:
        raise SystemExit("No transcript entries found.")
    by_day = common.group_by_day(entries)
    eff_labels_dir = labels_dir or (os.path.join(out_dir, "labels") if out_dir else None)
    labels = common.load_labels(eff_labels_dir)
    if ai_insights_path and os.path.isfile(ai_insights_path):
        app_ai = common.read_json(ai_insights_path)
        daily_ai = (app_ai or {}).get("daily", app_ai) if isinstance(app_ai, dict) else {}
        if isinstance(daily_ai, dict):
            for day_k, rec in daily_ai.items():
                if day_k not in labels and isinstance(rec, dict):
                    labels[day_k] = {
                        "mood": rec.get("mood"),
                        "mood_score": rec.get("mood_score"),
                        "topics": rec.get("topics", []),
                        "activities": rec.get("activities", []),
                    }

    clusters_p = os.path.join(out_dir, "clusters.json") if out_dir else common.out_path("clusters.json")
    cats_p = os.path.join(out_dir, "categories.json") if out_dir else common.out_path("categories.json")
    clusters = common.read_json(clusters_p)
    cats = common.read_json(cats_p)
    phrase_cluster = (clusters or {}).get("phrase_cluster") or {}
    cluster_to_category = {str(k): v for k, v
                           in ((cats or {}).get("cluster_to_category") or {}).items()}
    category_order = list((cats or {}).get("category_order") or [])
    have_categories = bool(phrase_cluster and cluster_to_category and category_order)

    def category_counts(label):
        counts = collections.Counter()
        if not have_categories:
            return counts
        for t in label.get("topics", []):
            t = str(t).strip()
            cid = phrase_cluster.get(t)
            if cid is None:
                continue
            cat = cluster_to_category.get(str(cid))
            if cat:
                counts[cat] += 1
        return counts

    # ---- single-pass tokenization & per-day aggregation ----------------
    entry_word_counts = []
    by_day_stats = collections.defaultdict(lambda: {"words": 0, "de_words": 0, "entries": 0})
    for e in entries:
        toks = common.tokenize_words(e["text"])
        n = len(toks)
        entry_word_counts.append((e, n))
        day_key = e["timestamp"].strftime("%Y-%m-%d")
        st_day = by_day_stats[day_key]
        st_day["words"] += n
        st_day["entries"] += 1
        if n and common.detect_language(e["text"], toks) == "de":
            st_day["de_words"] += n

    daily = []
    for day in sorted(by_day):
        label = labels.get(day) or {}
        st_day = by_day_stats[day]
        words = st_day["words"]
        de_words = st_day["de_words"]
        de_share = round(de_words / words, 4) if words else 0.0
        daily.append({
            "date": day,
            "weekday": common.WEEKDAY_NAMES[dt.date.fromisoformat(day).weekday()],
            "words": words,
            "de_words": de_words,
            "entries": st_day["entries"],
            "mood": label.get("mood_score"),
            "mood_label": label.get("mood"),
            "german_share": de_share,
            "activities": label.get("activities", []),
            "categories": dict(category_counts(label)),
        })
    daily.sort(key=lambda d: d["date"])

    # ---- rolling 7-day mood mean ---------------------------------------
    window = collections.deque()
    for row in daily:
        if row["mood"] is not None:
            window.append((row["date"], row["mood"]))
        cutoff = (dt.date.fromisoformat(row["date"]) - dt.timedelta(days=6)).isoformat()
        while window and window[0][0] < cutoff:
            window.popleft()
        row["mood_roll7"] = round(st.mean(v for _, v in window), 1) if window else None

    # ---- period buckets -------------------------------------------------
    def bucket(rows, key):
        out = {}
        for row in rows:
            k = key(row)
            b = out.setdefault(k, {"words": 0, "de_words": 0, "entries": 0, "days": 0,
                                   "mood": [], "cats": collections.Counter(),
                                   "acts": collections.Counter()})
            b["words"] += row["words"]
            b["de_words"] += row["de_words"]
            b["entries"] += row["entries"]
            b["days"] += 1
            if row["mood"] is not None:
                b["mood"].append(row["mood"])
            b["cats"].update(row["categories"])
            b["acts"].update(row["activities"])
        return out

    def shares(counter):
        total = sum(counter.values()) or 1
        return {c: round(counter.get(c, 0) / total, 4) for c in category_order}

    def emit(buckets, label_key, extra=None):
        rows = []
        for k in sorted(buckets):
            b = buckets[k]
            row = {
                label_key: k,
                "days": b["days"],
                "words": b["words"],
                "entries": b["entries"],
                "mood_mean": mean(b["mood"]),
                "mood_days": len(b["mood"]),
                "german_share": round(b["de_words"] / b["words"], 3) if b["words"] else 0.0,
                "activities": dict(b["acts"]),
            }
            if have_categories:
                row["category_counts"] = dict(b["cats"])
                row["category_share"] = shares(b["cats"])
            if extra:
                row.update(extra(k))
            rows.append(row)
        return rows

    months = emit(bucket(daily, lambda r: r["date"][:7]), "month")
    weeks = emit(bucket(daily, lambda r: common.iso_week(r["date"])), "week",
                 extra=lambda w: {"week_start": common.week_start(w)})

    # ---- weekday and hour profiles --------------------------------------
    wd_words = collections.Counter()
    wd_entries = collections.Counter()
    wd_days = collections.Counter()
    wd_mood = collections.defaultdict(list)
    for row in daily:
        wd_words[row["weekday"]] += row["words"]
        wd_entries[row["weekday"]] += row["entries"]
        wd_days[row["weekday"]] += 1
        if row["mood"] is not None:
            wd_mood[row["weekday"]].append(row["mood"])
    weekdays = [{
        "weekday": n,
        "words": wd_words[n],
        "entries": wd_entries[n],
        "days": wd_days[n],
        "mood_days": len(wd_mood[n]),
        "mood_mean": mean(wd_mood[n]),
    } for n in common.WEEKDAY_NAMES]

    hour_words = collections.Counter()
    hour_entries = collections.Counter()
    hour_mood_num = collections.Counter()
    hour_mood_den = collections.Counter()
    grid = collections.Counter()  # (weekday index, hour) -> words
    mood_by_day = {r["date"]: r["mood"] for r in daily}
    for e, n in entry_word_counts:
        ts = e["timestamp"]
        hour_words[ts.hour] += n
        hour_entries[ts.hour] += 1
        grid[(ts.weekday(), ts.hour)] += n
        mood = mood_by_day.get(ts.strftime("%Y-%m-%d"))
        if mood is not None and n:
            hour_mood_num[ts.hour] += mood * n
            hour_mood_den[ts.hour] += n
    hours = [{"hour": h, "words": hour_words[h], "entries": hour_entries[h],
              "mood_mean": (round(hour_mood_num[h] / hour_mood_den[h], 1)
                            if hour_mood_den[h] else None)}
             for h in range(24)]
    heat = [[grid[(w, h)] for h in range(24)] for w in range(7)]

    total_words = sum(r["words"] for r in daily)
    total_de_words = sum(r.pop("de_words", 0) for r in daily)
    busiest = max(range(24), key=lambda h: hour_words[h]) if total_words else None
    moods = [r["mood"] for r in daily if r["mood"] is not None]

    # ---- longitudinal lexical & pause health proxies (100% local) -------
    lexical_block = None
    try:
        import sys as _sys
        if common.APP_DIR not in _sys.path:
            _sys.path.insert(0, common.APP_DIR)
        import lexical_insights as _li

        tx_dir = transcripts_dir or common.TRANSCRIPTIONS_DIR
        cache_p = (
            cache_path
            or (os.path.join(out_dir, "lexical_series_cache.json") if out_dir else None)
            or common.out_path("lexical_series_cache.json")
        )
        idx = _li.LexicalDiskIndex(transcriptions_dir=tx_dir, cache_path=cache_p)
        idx.refresh(persist=True)
        last_d = dt.date.fromisoformat(daily[-1]["date"])
        do_trim = bool(trim_outliers)
        sl_all = _li.query_slice(idx, "All", "Daily", "All", today=last_d, trim_outliers=do_trim)
        sl_6m = _li.query_slice(idx, "6 months", "Daily", "All", today=last_d, trim_outliers=do_trim)
        sl_ytd = _li.query_slice(idx, "This year", "Daily", "All", today=last_d, trim_outliers=do_trim)
        report_keys = (
            "clarity_index",
            "mattr_diversity",
            "pause_long_pct",
            "vague_per_100",
            "vitality_balance",
        )
        lexical_block = {
            "trim_outliers": do_trim,
            "callout": sl_6m["callout"],
            "baselines": sl_all["baselines"],
            "summary_all": sl_all["summary"],
            "summary_6m": sl_6m["summary"],
            "summary_ytd": sl_ytd["summary"],
            "points": sl_all["points"],
            "rolling": {
                k: sl_all["rolling"].get(k, [])
                for k in (
                    "clarity_index",
                    "mattr_diversity",
                    "vague_per_100",
                    "pause_long_pct",
                    "fillers_per_100",
                    "self_focus_per_100",
                    "absolutist_per_100",
                    "vitality_balance",
                    "late_night_pct",
                )
            },
            "interpretations": {
                k: _li.interpret_metric_window(sl_all, k) for k in report_keys
            },
        }
    except Exception:
        lexical_block = None

    return {
        "generated": dt.date.today().isoformat(),
        "has_categories": have_categories,
        "categories": category_order,
        "category_definitions": (cats or {}).get("definitions", {}),
        "activities": common.ACTIVITIES,
        "headline": {
            "first_day": daily[0]["date"],
            "last_day": daily[-1]["date"],
            "total_words": total_words,
            "total_entries": sum(r["entries"] for r in daily),
            "days_active": len(daily),
            "days_labelled": len(moods),
            "longest_streak": longest_streak([r["date"] for r in daily]),
            "busiest_hour": busiest,
            "mood_mean": mean(moods),
            "german_share": round(total_de_words / total_words, 3) if total_words else 0.0,
            "words_per_active_day": round(total_words / len(daily)) if daily else 0,
        },
        "daily": daily,
        "months": months,
        "weeks": weeks,
        "weekdays": weekdays,
        "hours": hours,
        "heat_weekday_hour": heat,
        "lexical": lexical_block,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Build out/series.json from labels, clusters and transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transcripts", metavar="DIR", help="transcript archive directory")
    p.add_argument("--labels", metavar="DIR", help="labels directory")
    p.add_argument(
        "--trim-outliers",
        action="store_true",
        help="exclude >2.5σ outlier spikes non-destructively from lexical series",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    series = build(args.transcripts, args.labels, trim_outliers=args.trim_outliers)
    path = common.out_path("series.json")
    common.write_json(path, series)
    h = series["headline"]
    print(f"{h['first_day']} .. {h['last_day']}  "
          f"{h['total_words']:,} words, {h['total_entries']:,} entries, "
          f"{h['days_active']} active days, longest streak {h['longest_streak']}")
    print(f"labelled days {h['days_labelled']}, mean mood {h['mood_mean']}, "
          f"German share {h['german_share']}, busiest hour {h['busiest_hour']}")
    if not series["has_categories"]:
        print("no clusters found - category shares omitted "
              "(run cluster_topics.py to add them)")
    header = "month  days  words  mood  de%"
    if series["has_categories"]:
        header += "  " + " ".join(c[:8] for c in series["categories"])
    print(header)
    for m in series["months"]:
        line = (f"{m['month']} {m['days']:5d} {m['words']:7d} "
                f"{str(m['mood_mean']):>5s} {str(m['german_share']):>5s}")
        if series["has_categories"]:
            line += "  " + " ".join(
                f"{int(100 * m['category_share'][c]):3d}" for c in series["categories"])
        print(line)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
