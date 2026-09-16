# Analyse your own dictation data

This toolkit turns the transcript archive the app already writes into a single
self-contained HTML report: how much you spoke, when, in which language, what
about, and how your days felt.

It reads `transcriptions/*.txt` (and optionally the app's `ai_insights.json`),
writes everything to `analysis/out/`, and touches nothing else.

Example of what comes out:

- headline stats (words, dictations, active days, longest streak, busiest hour)
- words per month
- mood over time with a 7-day rolling mean
- what you talked about, clustered into your own categories, month by month
- a weekday x hour heat grid of when you actually speak
- German share per month, mood by weekday
- mood vs meetings, if you feed it a calendar export

---

## Privacy: what leaves your machine

| Step | Network | What is sent |
| --- | --- | --- |
| `common.py` | none | - |
| `label_days.py` | **OpenAI** | one day's transcript text per request (truncated to `--max-chars`) |
| `label_days.py --from-app-insights` | none | - |
| `cluster_topics.py` | **OpenAI** | only the short topic phrases from your labels, plus one naming request; never raw transcripts |
| `build_series.py` | none | - |
| `calendar_features.py` | none | - |
| `report.py` | none | - |

The report itself has no CDN links, no remote fonts and no JavaScript
libraries, so opening it makes no network requests either.

If you would rather send nothing at all: run
`label_days.py --from-app-insights` (uses the labels the app already produced),
skip `cluster_topics.py`, and everything else still works - the report simply
omits the category-mix section.

---

## Setup

Install the dependencies into the same environment you run the app from:

```
pip install -r analysis/requirements.txt
```

Python 3.11 or newer. Dependencies: `openai`, `numpy`, `scikit-learn`. The
calendar step uses the standard library only.

**API key.** The AI steps resolve the key in this order, and never print it:

1. the `OPENAI_API_KEY` environment variable
2. the app's per-machine config overlay `whisper_config.<HOSTNAME>.json`
3. the app's shared `whisper_config.json`

If you have already configured an OpenAI key in the app, there is nothing to do.
Check what the toolkit sees with:

```
python analysis/common.py
```

That prints the archive size, the date range, the output directory and *where*
the key came from (never the key).

---

## Run order

```
# 1. label each day (OpenAI; or use --from-app-insights for a free bootstrap)
python analysis/label_days.py --dry-run          # cost estimate, no calls
python analysis/label_days.py

# 2. cluster the topic phrases into your own categories (OpenAI, cheap)
python analysis/cluster_topics.py

# 3. build the series file (local)
python analysis/build_series.py

# 4. optional: join a calendar export (local)
python analysis/calendar_features.py --ics /path/to/export.ics

# 5. render the report (local)
python analysis/report.py
```

Or run the whole thing:

```
python analysis/run_all.py --label --cluster --ics /path/to/export.ics
```

`run_all.py` with no flags runs only the local steps, which is what you want
after the labels already exist.

Every script supports `--help`.

### Useful flags

- `label_days.py --since 2026-01-01 --until 2026-06-30` - label a slice
- `label_days.py --limit 5` - try a handful of days first
- `label_days.py --model gpt-4o` - better labels, ~15x the cost
- `label_days.py --min-words 60` - skip near-empty days (default)
- `label_days.py --overwrite` - re-label days you already have
- `cluster_topics.py --k 12` - fix the cluster count instead of searching
- `cluster_topics.py --no-name` - skip the naming call, numeric clusters only

---

## Rough cost

For a year of daily dictation (~350 labelled days, ~2,000 words per day) with
the default `gpt-4o-mini`:

| Step | Calls | Rough cost |
| --- | --- | --- |
| `label_days.py` | 1 per day (~350) | **$0.30 - $0.60** |
| `cluster_topics.py` embeddings | ~3,000 phrases | **under $0.01** |
| `cluster_topics.py` naming | 1 | **under $0.01** |
| everything else | 0 | free |

`label_days.py` prints its own estimate before it starts and the actual token
cost when it finishes. With `--model gpt-4o` the labelling step is roughly
$5-$10 for the same year. Prices are hard-coded in `label_days.py` and
`cluster_topics.py`; update them there if OpenAI's rates change.

Both AI steps cache to disk and skip work that is already done, so an
interrupted run costs nothing extra to resume.

---

## Output files (`analysis/out/`)

| File | Contents |
| --- | --- |
| `labels/YYYY-MM-DD.json` | one label per day: `mood`, `mood_score` (0-100), `daily_brief`, `topics`, `activities`, `language_share_german` |
| `embeddings.json` | cached embedding vectors for every topic phrase (so re-clustering is free) |
| `clusters.json` | k, each cluster's central and frequent phrases, the phrase to cluster map, per-month cluster weights |
| `categories.json` | cluster names, cluster to category map, category order, one-line category definitions |
| `cluster_report.txt` | the same clustering as readable text - skim this to sanity-check the categories |
| `series.json` | everything the report needs: headline stats, per-day rows (words, entries, mood, rolling mood, German share), per-month and per-week aggregates, weekday and hour profiles, weekday x hour word grid |
| `calendar_features.json` | per-day meeting count, meeting hours, all-day and off/holiday flags, talk and travel flags, plus the mood cross-tables |
| `report.html` | the finished single-file report |

`analysis/out/` is gitignored. Nothing is ever written outside it.

The fixed activity vocabulary is `coding, writing, client work, strategy,
research, personal, admin` (defined once in `common.py`). Change it there
before you label if it does not fit how you work - the labels, series and
report all read it from the same place.

---

## Adding your own calendar

**Google Calendar:** Settings -> Import & export -> Export. You get a zip with
one `.ics` per calendar. Unzip it and point the script at the one you want:

```
python analysis/calendar_features.py --ics ~/Downloads/work.ics
```

**Anything else:** produce a JSON list and use `--json`:

```json
[
  {"start": "2026-03-04T09:00:00", "end": "2026-03-04T10:00:00",
   "allDay": false, "summary": "Weekly sync"},
  {"start": "2026-03-09", "end": "2026-03-16", "allDay": true,
   "summary": "Vacation"}
]
```

The script derives, per day: meeting count, meeting hours, an all-day flag, an
off/holiday flag, and talk/travel flags from a generic keyword regex (English
and German). Those regexes live at the top of `calendar_features.py` - edit
them to match how you name things.

Then re-render: `python analysis/report.py` picks up
`out/calendar_features.json` automatically and adds the mood-vs-meetings
section.

Two limitations worth knowing: repeating events are expanded with a minimal
`RRULE` reader (`FREQ`, `INTERVAL`, `COUNT`, `UNTIL`, and `BYDAY` for weekly
rules), and time zones in the export are read as local time. Both are fine for
"how many meetings did I have that day"; neither is precise enough for billing.

---

## Reading the results honestly

- **Mood scores are a model's guess** at the tone of what you dictated, not a
  measurement of how you felt. Compare them to each other, never to an absolute
  scale, and never across models - re-labelling with a different `--model`
  shifts the whole curve.
- **Correlations on a few hundred days are weak evidence.** The mood-vs-meetings
  table is a hint. Treat anything under |r| = 0.2 as noise.
- **Language skews everything keyword-based.** If some months are heavier in one
  language, raw keyword counts drop for reasons that have nothing to do with
  your work. That is why this toolkit reports category *shares* rather than
  rates, and why the topic clustering runs on embeddings, not word lists.
- **Cluster only concrete topics.** The labelling prompt insists on concrete
  noun phrases for exactly this reason: abstract themes ("productivity",
  "communication") all embed near each other and collapse into one useless
  cluster. If your `cluster_report.txt` looks mushy, that is usually the cause.
- **Check `cluster_report.txt` before believing the category chart.** It takes a
  minute and it is the only step where a bad automatic name can quietly distort
  the story.
- **Days you did not dictate are missing, not zero.** A quiet week may mean a
  holiday, or it may mean you typed instead.
