"""Optional: join a calendar export with the day labels. Fully local.

Input is either an ``.ics`` file (Google Calendar -> Settings -> Import & export
-> Export, then unzip and point at one .ics) or a JSON list of
``{"start": ..., "end": ..., "allDay": true/false, "summary": ...}``.

Writes ``analysis/out/calendar_features.json`` with per-day meeting counts,
meeting hours, an all-day/holiday flag and generic talk/travel flags, plus
mood-vs-meetings and mood-by-weekday tables. Nothing is sent anywhere.

Examples::

    python analysis/calendar_features.py --ics ~/Downloads/mycalendar.ics
    python analysis/calendar_features.py --json events.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import re
import statistics as st

import common

# Generic keyword flags. Deliberately broad and language-mixed; edit freely.
TALK_RE = re.compile(
    r"\b(keynote|talk|webinar|masterclass|podcast|panel|speaker|conference|"
    r"congress|summit|workshop|seminar|lecture|vortrag|konferenz|schulung)\b", re.I)
TRAVEL_RE = re.compile(
    r"\b(flight|fly|train|travel|trip|commute|drive|airport|hotel|onsite|"
    r"on-site|flug|zug|reise|bahn|anreise|abreise|fahrt)\b", re.I)
OFF_RE = re.compile(
    r"\b(holiday|public holiday|bank holiday|vacation|pto|day off|off\b|ooo|"
    r"out of office|leave|feiertag|urlaub|ferien|frei)\b", re.I)


# --------------------------------------------------------------------------
# ICS parsing (stdlib only)
# --------------------------------------------------------------------------

def _unfold(text):
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _unescape(value):
    return (value.replace("\\n", " ").replace("\\N", " ")
                 .replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\"))


def _parse_dt(value, params):
    """Return (datetime_or_date, is_all_day). Timezones are treated as local."""
    value = value.strip()
    if params.get("VALUE") == "DATE" or (len(value) == 8 and value.isdigit()):
        return dt.date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}"), True
    value = value.rstrip("Z")
    try:
        return dt.datetime.strptime(value, "%Y%m%dT%H%M%S"), False
    except ValueError:
        return None, False


_WEEKDAY_CODES = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def _expand_rrule(start, rule, horizon, cap=1500):
    """Minimal RRULE expansion: FREQ/INTERVAL/COUNT/UNTIL/BYDAY(weekly)."""
    parts = {}
    for chunk in rule.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.upper()] = v
    freq = parts.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        return [start]
    interval = max(1, int(parts.get("INTERVAL", "1") or 1))
    count = int(parts["COUNT"]) if parts.get("COUNT", "").isdigit() else None
    until = None
    if parts.get("UNTIL"):
        parsed, _ = _parse_dt(parts["UNTIL"], {})
        if isinstance(parsed, dt.datetime):
            until = parsed.date()
        elif isinstance(parsed, dt.date):
            until = parsed
    bydays = [_WEEKDAY_CODES[d[-2:].upper()] for d in parts.get("BYDAY", "").split(",")
              if d and d[-2:].upper() in _WEEKDAY_CODES]

    base_date = start.date() if isinstance(start, dt.datetime) else start
    out = []
    cursor = base_date
    guard = 0
    while len(out) < (count or cap) and guard < cap * 4:
        guard += 1
        if cursor > horizon or (until and cursor > until):
            break
        if freq == "WEEKLY" and bydays:
            week_start = cursor - dt.timedelta(days=cursor.weekday())
            for wd in sorted(bydays):
                day = week_start + dt.timedelta(days=wd)
                if day < base_date or day > horizon or (until and day > until):
                    continue
                out.append(day)
                if count and len(out) >= count:
                    break
            cursor = cursor + dt.timedelta(weeks=interval)
            continue
        out.append(cursor)
        if freq == "DAILY":
            cursor = cursor + dt.timedelta(days=interval)
        elif freq == "WEEKLY":
            cursor = cursor + dt.timedelta(weeks=interval)
        elif freq == "MONTHLY":
            month = cursor.month - 1 + interval
            year = cursor.year + month // 12
            month = month % 12 + 1
            day = min(cursor.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0)
                                   else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
            cursor = dt.date(year, month, day)
        else:
            try:
                cursor = cursor.replace(year=cursor.year + interval)
            except ValueError:
                cursor = cursor.replace(year=cursor.year + interval, day=28)
    # rebuild datetimes at the original time of day
    if isinstance(start, dt.datetime):
        return [dt.datetime.combine(d, start.time()) for d in out]
    return out


def parse_ics(path, horizon=None):
    horizon = horizon or (dt.date.today() + dt.timedelta(days=1))
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = _unfold(fh.read())
    events = []
    cur = None
    for line in lines:
        if line.startswith("BEGIN:VEVENT"):
            cur = {}
            continue
        if line.startswith("END:VEVENT"):
            if cur is not None:
                events.extend(_finish_event(cur, horizon))
            cur = None
            continue
        if cur is None or ":" not in line:
            continue
        head, value = line.split(":", 1)
        bits = head.split(";")
        name = bits[0].upper()
        params = {}
        for b in bits[1:]:
            if "=" in b:
                pk, pv = b.split("=", 1)
                params[pk.upper()] = pv
        if name in ("DTSTART", "DTEND"):
            cur[name] = _parse_dt(value, params)
        elif name in ("SUMMARY", "RRULE", "STATUS", "TRANSP"):
            cur[name] = value
        elif name == "EXDATE":
            parsed, _ = _parse_dt(value.split(",")[0], params)
            cur.setdefault("EXDATE", []).append(parsed)
    return events


def _finish_event(cur, horizon):
    if "DTSTART" not in cur or cur["DTSTART"][0] is None:
        return []
    if str(cur.get("STATUS", "")).upper() == "CANCELLED":
        return []
    start, all_day = cur["DTSTART"]
    end, _ = cur.get("DTEND", (None, all_day))
    if end is None:
        end = (start + dt.timedelta(days=1)) if all_day else (start + dt.timedelta(hours=1))
    duration = end - start
    summary = _unescape(cur.get("SUMMARY", ""))
    starts = ([start] if "RRULE" not in cur
              else _expand_rrule(start, cur["RRULE"], horizon))
    excluded = {x for x in cur.get("EXDATE", []) if x is not None}
    out = []
    for s in starts:
        if s in excluded:
            continue
        out.append({
            "start": s.isoformat(),
            "end": (s + duration).isoformat(),
            "allDay": all_day,
            "summary": summary,
        })
    return out


def load_json_events(path):
    data = common.read_json(path)
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list of events")
    out = []
    for e in data:
        if not isinstance(e, dict) or not e.get("start"):
            continue
        out.append({
            "start": str(e["start"]),
            "end": str(e.get("end") or e["start"]),
            "allDay": bool(e.get("allDay")),
            "summary": str(e.get("summary") or ""),
        })
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _days_of(event):
    s = event["start"][:10]
    if not event["allDay"]:
        return [s]
    try:
        d0 = dt.date.fromisoformat(s)
        d1 = dt.date.fromisoformat(event["end"][:10])
    except ValueError:
        return [s]
    span = max(1, (d1 - d0).days)  # DTEND is exclusive for all-day events
    return [(d0 + dt.timedelta(i)).isoformat() for i in range(span)]


def _hours(event):
    if event["allDay"]:
        return 0.0
    try:
        a = dt.datetime.fromisoformat(event["start"])
        b = dt.datetime.fromisoformat(event["end"])
    except ValueError:
        return 0.0
    return max(0.0, (b - a).total_seconds() / 3600)


def build_features(events, min_minutes=10):
    feats = collections.defaultdict(
        lambda: {"meetings": 0, "meeting_hours": 0.0, "all_day": False,
                 "off_or_holiday": False, "talk": False, "travel": False,
                 "all_day_titles": []})
    for e in events:
        summary = e.get("summary", "")
        for day in _days_of(e):
            f = feats[day]
            if e["allDay"]:
                f["all_day"] = True
                if summary:
                    f["all_day_titles"].append(summary)
                if OFF_RE.search(summary):
                    f["off_or_holiday"] = True
            else:
                h = _hours(e)
                if h * 60 >= min_minutes:
                    f["meetings"] += 1
                    f["meeting_hours"] += h
            if TALK_RE.search(summary):
                f["talk"] = True
            if TRAVEL_RE.search(summary):
                f["travel"] = True
    for f in feats.values():
        f["meeting_hours"] = round(f["meeting_hours"], 2)
    return dict(sorted(feats.items()))


def _pearson(pairs):
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = st.mean(xs), st.mean(ys)
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    if den == 0:
        return None
    return round(sum((a - mx) * (b - my) for a, b in pairs) / den, 3)


def build_tables(feats, moods):
    def group(name, predicate):
        xs = [moods[d] for d in moods if predicate(d)]
        return {"group": name, "n": len(xs),
                "mood_mean": round(st.mean(xs), 1) if xs else None,
                "mood_median": round(st.median(xs), 1) if xs else None}

    def f(day):
        return feats.get(day, {"meetings": 0, "meeting_hours": 0.0,
                               "off_or_holiday": False, "talk": False,
                               "travel": False, "all_day": False})

    wd = lambda d: dt.date.fromisoformat(d).weekday()
    meetings = [
        group("all days", lambda d: True),
        group("0 meetings", lambda d: f(d)["meetings"] == 0),
        group("1-2 meetings", lambda d: 1 <= f(d)["meetings"] <= 2),
        group("3-4 meetings", lambda d: 3 <= f(d)["meetings"] <= 4),
        group("5+ meetings", lambda d: f(d)["meetings"] >= 5),
        group("3h+ in meetings", lambda d: f(d)["meeting_hours"] >= 3),
        group("off / holiday", lambda d: f(d)["off_or_holiday"]),
        group("talk or event", lambda d: f(d)["talk"]),
        group("travel", lambda d: f(d)["travel"]),
    ]
    weekday = [group(name, lambda d, i=i: wd(d) == i)
               for i, name in enumerate(common.WEEKDAY_NAMES)]
    r_all = _pearson([(f(d)["meetings"], moods[d]) for d in moods])
    r_week = _pearson([(f(d)["meetings"], moods[d]) for d in moods if wd(d) < 5])
    return {"mood_by_meetings": meetings, "mood_by_weekday": weekday,
            "pearson_meetings_mood": r_all,
            "pearson_meetings_mood_weekdays_only": r_week}


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'group':<18s} {'n':>4s} {'mean':>6s} {'median':>7s}")
    for r in rows:
        mean = "-" if r["mood_mean"] is None else f"{r['mood_mean']:.1f}"
        med = "-" if r["mood_median"] is None else f"{r['mood_median']:.1f}"
        print(f"  {r['group']:<18s} {r['n']:>4d} {mean:>6s} {med:>7s}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Per-day calendar features and mood cross-tables (local only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ics", metavar="FILE", help="calendar export (.ics)")
    src.add_argument("--json", metavar="FILE",
                     help="JSON list of {start,end,allDay,summary}")
    p.add_argument("--labels", metavar="DIR", help="labels directory")
    p.add_argument("--min-minutes", type=int, default=10,
                   help="ignore timed events shorter than this as meetings")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    events = parse_ics(args.ics) if args.ics else load_json_events(args.json)
    print(f"events parsed: {len(events)}")
    if not events:
        raise SystemExit("No events found in the export.")
    feats = build_features(events, args.min_minutes)
    labels = common.load_labels(args.labels)
    moods = {d: v["mood_score"] for d, v in labels.items()
             if isinstance(v.get("mood_score"), int)}
    print(f"days with calendar data: {len(feats)}; days with a mood label: {len(moods)}")

    tables = build_tables(feats, moods) if moods else None
    if tables:
        print_table("Mood by meeting load", tables["mood_by_meetings"])
        print_table("Mood by weekday", tables["mood_by_weekday"])
        print(f"\n  pearson r (meetings, mood): {tables['pearson_meetings_mood']} "
              f"(weekdays only: {tables['pearson_meetings_mood_weekdays_only']})")
    else:
        print("No mood labels yet - tables skipped. Run label_days.py first.")

    path = common.out_path("calendar_features.json")
    common.write_json(path, {
        "generated": dt.date.today().isoformat(),
        "source": args.ics or args.json,
        "n_events": len(events),
        "daily": feats,
        "tables": tables or {},
    })
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
