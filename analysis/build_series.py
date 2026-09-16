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
import statistics as st

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


def build(transcripts_dir=None, labels_dir=None):
    entries = common.load_entries(transcripts_dir)
    if not entries:
        raise SystemExit("No transcript entries found.")
    by_day = common.group_by_day(entries)
    labels = common.load_labels(labels_dir)

    clusters = common.read_json(common.out_path("clusters.json"))
    cats = common.read_json(common.out_path("categories.json"))
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

    # ---- per-day aggregation -------------------------------------------
    daily = []
    for day, day_entries in by_day.items():
        label = labels.get(day) or {}
        words = sum(common.word_count(e["text"]) for e in day_entries)
        daily.append({
            "date": day,
            "weekday": common.WEEKDAY_NAMES[dt.date.fromisoformat(day).weekday()],
            "words": words,
            "entries": len(day_entries),
            "mood": label.get("mood_score"),
            "mood_label": label.get("mood"),
            "german_share": common.day_german_share(day_entries),
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
            b = out.setdefault(k, {"words": 0, "entries": 0, "days": 0,
                                   "mood": [], "de": [], "cats": collections.Counter(),
                                   "acts": collections.Counter()})
            b["words"] += row["words"]
            b["entries"] += row["entries"]
            b["days"] += 1
            if row["mood"] is not None:
                b["mood"].append(row["mood"])
            b["de"].append(row["german_share"])
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
                "german_share": round(st.mean(b["de"]), 3) if b["de"] else None,
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
    wd_mood = collections.defaultdict(list)
    for row in daily:
        wd_words[row["weekday"]] += row["words"]
        wd_entries[row["weekday"]] += row["entries"]
        if row["mood"] is not None:
            wd_mood[row["weekday"]].append(row["mood"])
    weekdays = [{"weekday": n, "words": wd_words[n], "entries": wd_entries[n],
                 "days": len(wd_mood[n]), "mood_mean": mean(wd_mood[n])}
                for n in common.WEEKDAY_NAMES]

    hour_words = collections.Counter()
    hour_entries = collections.Counter()
    hour_mood_num = collections.Counter()
    hour_mood_den = collections.Counter()
    grid = collections.Counter()  # (weekday index, hour) -> words
    mood_by_day = {r["date"]: r["mood"] for r in daily}
    for e in entries:
        ts = e["timestamp"]
        n = common.word_count(e["text"])
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
    busiest = max(range(24), key=lambda h: hour_words[h]) if total_words else None
    moods = [r["mood"] for r in daily if r["mood"] is not None]

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
            "german_share": round(st.mean(r["german_share"] for r in daily), 3),
            "words_per_active_day": round(total_words / len(daily)) if daily else 0,
        },
        "daily": daily,
        "months": months,
        "weeks": weeks,
        "weekdays": weekdays,
        "hours": hours,
        "heat_weekday_hour": heat,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Build out/series.json from labels, clusters and transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transcripts", metavar="DIR", help="transcript archive directory")
    p.add_argument("--labels", metavar="DIR", help="labels directory")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    series = build(args.transcripts, args.labels)
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
