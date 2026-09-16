"""Render ``analysis/out/report.html`` from the series file. Fully local.

One self-contained page: inline CSS, inline SVG charts, no CDN, no JavaScript
libraries. Works in light and dark via ``prefers-color-scheme``.

Example::

    python analysis/report.py
    python analysis/report.py --title "My dictation year"
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import os

import common

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#8cd17d"]

CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfaf8;
  --panel: #ffffff;
  --ink: #1d2129;
  --ink-dim: #5b6472;
  --line: #e3e1dc;
  --accent: #4e79a7;
  --grid: #ecebe7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a;
    --panel: #1b1e24;
    --ink: #e8eaee;
    --ink-dim: #99a1ae;
    --line: #2c313a;
    --accent: #7aa9d8;
    --grid: #262b33;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 20px 64px;
  background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
}
main { max-width: 980px; margin: 0 auto; }
header { padding: 48px 0 8px; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 0 0 4px; letter-spacing: -0.005em; }
p.sub { color: var(--ink-dim); margin: 0 0 24px; }
section {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px 18px; margin: 0 0 20px;
}
section > p.note { color: var(--ink-dim); font-size: 13px; margin: 2px 0 14px; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 0; }
.stat {
  flex: 1 1 150px; border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; background: var(--bg);
}
.stat .v { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
.stat .k { color: var(--ink-dim); font-size: 12px; text-transform: uppercase;
           letter-spacing: 0.06em; }
.chart { width: 100%; overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 18px; margin: 12px 0 0;
          font-size: 13px; color: var(--ink-dim); }
.legend span.sw { display: inline-block; width: 11px; height: 11px;
                  border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 6px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-dim); font-weight: 500; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.05em; }
footer { color: var(--ink-dim); font-size: 13px; text-align: center; padding: 8px 0; }
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else ("-" if n is None else str(n))


def axis_text(x, y, text, anchor="middle", size=11, dim=True):
    fill = "var(--ink-dim)" if dim else "var(--ink)"
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="{fill}">{esc(text)}</text>')


def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">')


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def chart_bars(rows, key, value_key, label, unit="", height=220, color=None,
               max_override=None, value_fmt=None):
    """Simple vertical bar chart over ordered rows."""
    if not rows:
        return "<p class='note'>No data.</p>"
    w, pad_l, pad_r, pad_t, pad_b = 900, 56, 12, 14, 40
    plot_w = w - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vals = [(r.get(value_key) or 0) for r in rows]
    vmax = max_override or max(vals) or 1
    slot = plot_w / len(rows)
    bar_w = max(3.0, min(46.0, slot * 0.66))
    color = color or "var(--accent)"
    out = [svg_open(w, height, label)]
    for i in range(5):
        y = pad_t + plot_h * i / 4
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(axis_text(pad_l - 8, y + 4, fmt(int(round(vmax * (4 - i) / 4))),
                             anchor="end"))
    for i, r in enumerate(rows):
        v = r.get(value_key) or 0
        h = plot_h * v / vmax
        x = pad_l + slot * i + (slot - bar_w) / 2
        y = pad_t + plot_h - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                   f'height="{max(0.0, h):.1f}" rx="2" fill="{color}"><title>'
                   f'{esc(r.get(key))}: {esc((value_fmt or fmt)(v))}{esc(unit)}'
                   f'</title></rect>')
        step = max(1, len(rows) // 18)
        if i % step == 0:
            out.append(axis_text(x + bar_w / 2, height - pad_b + 18, r.get(key)))
    out.append("</svg>")
    return "".join(out)


def chart_mood(daily, height=250):
    pts = [r for r in daily if r.get("mood") is not None]
    if len(pts) < 2:
        return "<p class='note'>Not enough labelled days to plot mood.</p>"
    w, pad_l, pad_r, pad_t, pad_b = 900, 40, 12, 14, 34
    plot_w, plot_h = w - pad_l - pad_r, height - pad_t - pad_b
    first = dt.date.fromisoformat(pts[0]["date"])
    last = dt.date.fromisoformat(pts[-1]["date"])
    span = max(1, (last - first).days)

    def x_of(day):
        return pad_l + plot_w * (dt.date.fromisoformat(day) - first).days / span

    def y_of(v):
        return pad_t + plot_h * (100 - v) / 100

    out = [svg_open(w, height, "Mood over time")]
    for v in (0, 25, 50, 75, 100):
        y = y_of(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(axis_text(pad_l - 8, y + 4, v, anchor="end"))
    for r in pts:
        out.append(f'<circle cx="{x_of(r["date"]):.1f}" cy="{y_of(r["mood"]):.1f}" '
                   f'r="2" fill="var(--ink-dim)" opacity="0.45"><title>'
                   f'{esc(r["date"])}: {esc(r["mood"])}'
                   f'{" - " + esc(r["mood_label"]) if r.get("mood_label") else ""}'
                   f'</title></circle>')
    roll = [r for r in daily if r.get("mood_roll7") is not None]
    if len(roll) > 1:
        d = " ".join(f'{"M" if i == 0 else "L"}{x_of(r["date"]):.1f},'
                     f'{y_of(r["mood_roll7"]):.1f}' for i, r in enumerate(roll))
        out.append(f'<path d="{d}" fill="none" stroke="var(--accent)" '
                   f'stroke-width="2.2" stroke-linejoin="round"/>')
    months = sorted({r["date"][:7] for r in pts})
    for m in months:
        day = f"{m}-01"
        if day < pts[0]["date"]:
            day = pts[0]["date"]
        out.append(axis_text(x_of(day), height - pad_b + 20, m))
    out.append("</svg>")
    return "".join(out)


def chart_stacked(months, categories, height=280):
    if not months or not categories:
        return "<p class='note'>No categories - run cluster_topics.py.</p>"
    w, pad_l, pad_r, pad_t, pad_b = 900, 44, 12, 14, 40
    plot_w, plot_h = w - pad_l - pad_r, height - pad_t - pad_b
    slot = plot_w / len(months)
    bar_w = max(6.0, min(56.0, slot * 0.66))
    out = [svg_open(w, height, "Category mix per month")]
    for i in range(5):
        y = pad_t + plot_h * i / 4
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(axis_text(pad_l - 8, y + 4, f"{(4 - i) * 25}%", anchor="end"))
    for i, m in enumerate(months):
        share = m.get("category_share") or {}
        x = pad_l + slot * i + (slot - bar_w) / 2
        y = pad_t + plot_h
        for ci, cat in enumerate(categories):
            frac = share.get(cat, 0) or 0
            h = plot_h * frac
            if h <= 0:
                continue
            y -= h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                       f'height="{h:.1f}" fill="{PALETTE[ci % len(PALETTE)]}">'
                       f'<title>{esc(m["month"])} {esc(cat)}: '
                       f'{frac * 100:.0f}%</title></rect>')
        out.append(axis_text(x + bar_w / 2, height - pad_b + 18, m["month"]))
    out.append("</svg>")
    legend = "".join(
        f'<span><span class="sw" style="background:{PALETTE[i % len(PALETTE)]}"></span>'
        f'{esc(c)}</span>' for i, c in enumerate(categories))
    return "".join(out) + f'<div class="legend">{legend}</div>'


def chart_heat(grid, height=250):
    if not grid:
        return "<p class='note'>No data.</p>"
    w, pad_l, pad_r, pad_t, pad_b = 900, 46, 12, 20, 30
    cell_w = (w - pad_l - pad_r) / 24
    cell_h = (height - pad_t - pad_b) / 7
    vmax = max((max(row) for row in grid), default=0) or 1
    out = [svg_open(w, height, "Words by weekday and hour")]
    for wd in range(7):
        out.append(axis_text(pad_l - 8, pad_t + cell_h * wd + cell_h / 2 + 4,
                             common.WEEKDAY_NAMES[wd], anchor="end"))
        for h in range(24):
            v = grid[wd][h]
            alpha = 0.06 + 0.94 * (v / vmax) ** 0.6 if v else 0.0
            x = pad_l + cell_w * h
            y = pad_t + cell_h * wd
            fill = "var(--accent)" if v else "var(--grid)"
            out.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" '
                       f'width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" rx="2" '
                       f'fill="{fill}" opacity="{alpha if v else 1:.3f}">'
                       f'<title>{esc(common.WEEKDAY_NAMES[wd])} '
                       f'{h:02d}:00 - {esc(fmt(v))} words</title></rect>')
    for h in range(0, 24, 2):
        out.append(axis_text(pad_l + cell_w * h + cell_w / 2,
                             height - pad_b + 18, f"{h:02d}"))
    out.append("</svg>")
    return "".join(out)


def table_rows(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def render(series, calendar=None, title="Dictation report"):
    h = series["headline"]
    cats = series.get("categories") or []
    parts = []
    parts.append(f"<!doctype html>\n<html lang=\"en\">\n<head>\n"
                 f"<meta charset=\"utf-8\">\n"
                 f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                 f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>")
    parts.append(f"<header><h1>{esc(title)}</h1><p class='sub'>"
                 f"{esc(h['first_day'])} to {esc(h['last_day'])} &middot; generated "
                 f"{esc(series.get('generated', ''))}</p></header>")

    # 1 headline
    busiest = h.get("busiest_hour")
    stats = [
        ("Words spoken", fmt(h["total_words"])),
        ("Dictations", fmt(h["total_entries"])),
        ("Active days", fmt(h["days_active"])),
        ("Longest streak", f"{h['longest_streak']} days"),
        ("Busiest hour", "-" if busiest is None else f"{busiest:02d}:00"),
        ("Words / active day", fmt(h["words_per_active_day"])),
        ("Mean mood", "-" if h["mood_mean"] is None else f"{h['mood_mean']}"),
        ("German share", f"{round((h['german_share'] or 0) * 100)}%"),
    ]
    cards = "".join(f"<div class='stat'><div class='v'>{esc(v)}</div>"
                    f"<div class='k'>{esc(k)}</div></div>" for k, v in stats)
    parts.append(f"<section><h2>Headline</h2><p class='note'>"
                 f"{esc(h['days_labelled'])} of {esc(h['days_active'])} active days "
                 f"carry an AI label.</p><div class='stats'>{cards}</div></section>")

    # 2 words per month
    parts.append("<section><h2>Words per month</h2>"
                 "<p class='note'>Total words dictated, by calendar month.</p>"
                 f"<div class='chart'>"
                 f"{chart_bars(series['months'], 'month', 'words', 'Words per month')}"
                 f"</div></section>")

    # 3 mood over time
    parts.append("<section><h2>Mood over time</h2>"
                 "<p class='note'>One dot per labelled day (0-100); the line is a "
                 "7-day rolling mean.</p>"
                 f"<div class='chart'>{chart_mood(series['daily'])}</div></section>")

    # 4 category mix
    if series.get("has_categories"):
        parts.append("<section><h2>Category mix per month</h2>"
                     "<p class='note'>Share of the month's topic mentions per "
                     "category, stacked to 100%.</p>"
                     f"<div class='chart'>{chart_stacked(series['months'], cats)}</div>"
                     "</section>")

    # 5 weekly rhythm
    parts.append("<section><h2>Weekly rhythm</h2>"
                 "<p class='note'>Words dictated by weekday and hour of day.</p>"
                 f"<div class='chart'>{chart_heat(series['heat_weekday_hour'])}</div>"
                 "</section>")

    # 6 german share per month
    de_rows = [{"month": m["month"], "de": round((m["german_share"] or 0) * 100, 1)}
               for m in series["months"]]
    parts.append("<section><h2>German share per month</h2>"
                 "<p class='note'>Share of words detected as German, by the same "
                 "heuristic the app uses.</p>"
                 f"<div class='chart'>"
                 f"{chart_bars(de_rows, 'month', 'de', 'German share per month', unit='%', max_override=100, color='#b07aa1')}"
                 f"</div></section>")

    # 7 mood by weekday
    wd_rows = [r for r in series["weekdays"] if r["mood_mean"] is not None]
    if wd_rows:
        parts.append("<section><h2>Mood by weekday</h2>"
                     "<p class='note'>Mean labelled mood per weekday "
                     "(scale 0-100).</p>"
                     f"<div class='chart'>"
                     f"{chart_bars(series['weekdays'], 'weekday', 'mood_mean', 'Mood by weekday', max_override=100, color='#59a14f')}"
                     f"</div>"
                     + table_rows(["Weekday", "Days", "Mean mood", "Words"],
                                  [(r["weekday"], r["days"], r["mood_mean"],
                                    fmt(r["words"])) for r in series["weekdays"]])
                     + "</section>")

    # 8 mood vs meetings
    tables = (calendar or {}).get("tables") or {}
    if tables.get("mood_by_meetings"):
        rows = [(r["group"], r["n"],
                 "-" if r["mood_mean"] is None else r["mood_mean"],
                 "-" if r["mood_median"] is None else r["mood_median"])
                for r in tables["mood_by_meetings"]]
        r_all = tables.get("pearson_meetings_mood")
        r_wd = tables.get("pearson_meetings_mood_weekdays_only")
        bars = [{"group": r["group"], "mood_mean": r["mood_mean"] or 0}
                for r in tables["mood_by_meetings"]]
        parts.append("<section><h2>Mood vs meetings</h2>"
                     f"<p class='note'>From your calendar export. Pearson r "
                     f"(meetings, mood) = {esc(r_all)}; weekdays only "
                     f"{esc(r_wd)}. Small samples - read as a hint, not a "
                     f"finding.</p>"
                     f"<div class='chart'>"
                     f"{chart_bars(bars, 'group', 'mood_mean', 'Mood by meeting load', max_override=100, color='#e15759')}"
                     f"</div>"
                     + table_rows(["Group", "Days", "Mean mood", "Median"], rows)
                     + "</section>")

    parts.append(f"<footer>Generated locally by the dictation analysis toolkit "
                 f"on {esc(series.get('generated', ''))}. No data left this "
                 f"machine to build this page.</footer>")
    parts.append("</main>\n</body>\n</html>\n")
    return "\n".join(parts)


def build_parser():
    p = argparse.ArgumentParser(
        description="Render out/report.html from out/series.json (local only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--title", default="Dictation report", help="page title")
    p.add_argument("--series", metavar="FILE", help="path to series.json")
    p.add_argument("--calendar", metavar="FILE", help="path to calendar_features.json")
    p.add_argument("--output", metavar="FILE", help="output HTML path (inside out/)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    series_path = args.series or common.out_path("series.json")
    series = common.read_json(series_path)
    if not series:
        raise SystemExit(f"No series file at {series_path}. Run build_series.py first.")
    calendar = common.read_json(args.calendar or common.out_path("calendar_features.json"))
    out = args.output or common.out_path("report.html")
    if os.path.abspath(out).startswith(os.path.abspath(common.OUT_DIR)) is False:
        raise SystemExit("Refusing to write outside analysis/out/.")
    html_text = render(series, calendar, args.title)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    print(f"Wrote {out} ({len(html_text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
