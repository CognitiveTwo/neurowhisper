"""Label every dictation day with mood, a brief, topics and activities.

Sends one chat completion per day (the day's transcript text) to OpenAI and
caches the result in ``analysis/out/labels/YYYY-MM-DD.json``. Days that already
have a cached label are skipped, so the script is safe to re-run.

Privacy: this is the one step that uploads transcript text. Use
``--from-app-insights`` to build labels from the app's own ``ai_insights.json``
instead, which makes no API call at all.

Examples::

    python analysis/label_days.py --dry-run
    python analysis/label_days.py --since 2026-01-01 --limit 20
    python analysis/label_days.py --from-app-insights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import common

# Approximate USD per 1M tokens. Update if OpenAI changes prices; used only for
# the estimate printed before a run.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_TOKENS_PER_DAY = 400  # a label is ~300-500 tokens

SYSTEM_PROMPT = """You analyse one person's spoken dictation for a single day.
The text is a concatenation of everything they dictated that day, in the order
spoken. It may mix languages and it may be rough speech-to-text.

Return ONLY valid JSON with exactly this schema:
{
  "mood": "short phrase, 1-3 words, describing the day's overall tone",
  "mood_score": 50,
  "daily_brief": "3-5 sentences: what the person worked on, how the day went, and the underlying mindset. Neutral, factual, second-person-free.",
  "topics": ["concrete noun phrase", "concrete noun phrase", "concrete noun phrase"],
  "activities": ["coding"],
  "language_share_german": 0.0
}

Rules:
- mood_score is an integer 0-100 (0 = severely negative, 50 = neutral,
  100 = elated).
- topics: at least 3, ideally 5-10. Each must be a CONCRETE noun phrase naming
  a specific thing worked on or discussed ("invoice reminder email",
  "database migration script", "holiday booking"). Never abstract themes
  ("productivity", "communication", "growth") - those collapse into one blob
  when clustered later.
- activities: choose only from this fixed list, one or more, most prominent
  first: ACTIVITY_LIST
- language_share_german: fraction of the day's speech that is German, 0.0-1.0.
- Write mood, daily_brief and topics in English even when the speech is not.
- Output the JSON object and nothing else.""".replace(
    "ACTIVITY_LIST", ", ".join(common.ACTIVITIES)
)

REQUIRED_KEYS = ("mood", "mood_score", "daily_brief", "topics", "activities",
                 "language_share_german")


def validate(obj):
    """Return (ok, reason). Mutates obj in place to normalise light issues."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"missing keys: {', '.join(missing)}"
    try:
        obj["mood_score"] = int(round(float(obj["mood_score"])))
    except (TypeError, ValueError):
        return False, "mood_score is not a number"
    if not 0 <= obj["mood_score"] <= 100:
        return False, f"mood_score {obj['mood_score']} out of range 0-100"
    if not isinstance(obj["mood"], str) or not obj["mood"].strip():
        return False, "mood is empty"
    if not isinstance(obj["daily_brief"], str) or len(obj["daily_brief"].split()) < 10:
        return False, "daily_brief too short"
    topics = obj.get("topics")
    if not isinstance(topics, list):
        return False, "topics is not a list"
    topics = [str(t).strip() for t in topics if str(t).strip()]
    if len(topics) < 3:
        return False, f"only {len(topics)} topics, need at least 3"
    obj["topics"] = topics
    acts = obj.get("activities")
    if not isinstance(acts, list):
        return False, "activities is not a list"
    obj["activities"] = [a for a in (str(x).strip().lower() for x in acts)
                         if a in common.ACTIVITIES]
    try:
        share = float(obj["language_share_german"])
    except (TypeError, ValueError):
        return False, "language_share_german is not a number"
    obj["language_share_german"] = round(min(1.0, max(0.0, share)), 3)
    return True, ""


def label_one(client, model, day, text, temperature=0.2):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Date: {day}\n\n{text}"},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    usage = resp.usage
    content = resp.choices[0].message.content or ""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc})", usage
    ok, reason = validate(obj)
    return (obj if ok else None), ("" if ok else reason), usage


def from_app_insights(args, day_entries):
    """Bootstrap labels from the app's ai_insights.json - no API call."""
    path = args.app_insights or common.AI_INSIGHTS_FILE
    cache = common.read_json(path)
    if not isinstance(cache, dict) or not cache:
        raise SystemExit(f"No usable app insight cache at {path}")
    labels_dir = common.ensure_out("labels")
    written = skipped = unusable = 0
    for day, value in sorted(cache.items()):
        if args.since and day < args.since:
            continue
        if args.until and day > args.until:
            continue
        target = os.path.join(labels_dir, f"{day}.json")
        if os.path.exists(target) and not args.overwrite:
            skipped += 1
            continue
        if not isinstance(value, dict):
            unusable += 1
            continue
        topics = [str(t).strip() for t in
                  (list(value.get("projects") or []) + list(value.get("topics") or [])
                   + list(value.get("themes") or []))
                  if str(t).strip()]
        seen, uniq = set(), []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                uniq.append(t)
        label = {
            "mood": str(value.get("mood", "")).strip() or "unlabelled",
            "mood_score": value.get("mood_score", 50),
            "daily_brief": str(value.get("daily_brief", "")).strip(),
            "topics": uniq,
            "activities": [a for a in (str(x).strip().lower()
                                       for x in value.get("activities") or [])
                           if a in common.ACTIVITIES],
            "language_share_german": value.get(
                "language_share_german",
                common.day_german_share(day_entries.get(day, [])),
            ),
            "source": "app_insights",
        }
        ok, reason = validate(label)
        if not ok:
            unusable += 1
            if args.verbose:
                print(f"  skip {day}: {reason}")
            continue
        common.write_json(target, label, indent=1)
        written += 1
        if args.limit and written >= args.limit:
            break
    print(f"\nBootstrapped {written} label(s) from {os.path.basename(path)}; "
          f"{skipped} already cached, {unusable} unusable.")
    print(f"Labels: {labels_dir}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        description="Label dictation days with mood, brief, topics, activities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="OpenAI chat model used for labelling")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="first day to label")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="last day to label")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many new labels (0 = no limit)")
    p.add_argument("--max-chars", type=int, default=24000,
                   help="max characters of a day's transcript sent to the model")
    p.add_argument("--min-words", type=int, default=60,
                   help="skip days with fewer spoken words than this")
    p.add_argument("--from-app-insights", action="store_true",
                   help="build labels from the app's ai_insights.json, no API call")
    p.add_argument("--app-insights", metavar="PATH",
                   help="path to ai_insights.json (default: the app's own)")
    p.add_argument("--transcripts", metavar="DIR",
                   help="transcript archive directory (default: the app's own)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-label days that already have a cached label")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and cost estimate, then exit")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="seconds to pause between API calls")
    p.add_argument("--verbose", action="store_true", help="per-day logging")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    entries = common.load_entries(args.transcripts, since=args.since, until=args.until)
    by_day = common.group_by_day(entries)

    if args.from_app_insights:
        return from_app_insights(args, by_day)

    if not by_day:
        raise SystemExit("No transcript entries found. Dictate something first.")

    labels_dir = common.ensure_out("labels")
    todo = []
    for day, day_entries in by_day.items():
        target = os.path.join(labels_dir, f"{day}.json")
        if os.path.exists(target) and not args.overwrite:
            continue
        text = common.day_text(day_entries)
        if common.word_count(text) < args.min_words:
            continue
        todo.append((day, text[:args.max_chars]))
    if args.limit:
        todo = todo[:args.limit]

    in_price, out_price = PRICES.get(args.model, PRICES[DEFAULT_MODEL])
    est_in = sum(len(t) for _, t in todo) / 4 + len(todo) * len(SYSTEM_PROMPT) / 4
    est_out = len(todo) * OUTPUT_TOKENS_PER_DAY
    est_cost = est_in / 1e6 * in_price + est_out / 1e6 * out_price
    cached = sum(1 for day in by_day
                 if os.path.exists(os.path.join(labels_dir, f"{day}.json")))
    print(f"Days with speech      : {len(by_day)}")
    print(f"Already labelled      : {cached}")
    print(f"To label now          : {len(todo)}")
    print(f"Model                 : {args.model}")
    print(f"Estimated tokens      : ~{int(est_in):,} in / ~{int(est_out):,} out")
    print(f"Estimated cost        : ~${est_cost:.2f} (prices as configured in this script)")
    if not todo:
        print("Nothing to do.")
        return 0
    if args.dry_run:
        print("Dry run - no API calls made.")
        return 0

    key = common.require_api_key()
    print(f"API key source        : {common.key_source()}")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("The openai package is missing: pip install -r analysis/requirements.txt")
    client = OpenAI(api_key=key)

    done = failed = 0
    tok_in = tok_out = 0
    for i, (day, text) in enumerate(todo, 1):
        obj, reason, usage = label_one(client, args.model, day, text)
        if usage is not None:
            tok_in += getattr(usage, "prompt_tokens", 0) or 0
            tok_out += getattr(usage, "completion_tokens", 0) or 0
        if obj is None:
            if args.verbose:
                print(f"  [{i}/{len(todo)}] {day}: {reason} - retrying once")
            obj, reason, usage = label_one(client, args.model, day, text, temperature=0.0)
            if usage is not None:
                tok_in += getattr(usage, "prompt_tokens", 0) or 0
                tok_out += getattr(usage, "completion_tokens", 0) or 0
        if obj is None:
            failed += 1
            print(f"  [{i}/{len(todo)}] {day}: FAILED ({reason})")
            continue
        obj["model"] = args.model
        common.write_json(os.path.join(labels_dir, f"{day}.json"), obj, indent=1)
        done += 1
        if args.verbose:
            print(f"  [{i}/{len(todo)}] {day}: {obj['mood']} ({obj['mood_score']}), "
                  f"{len(obj['topics'])} topics")
        else:
            sys.stdout.write(f"\r  labelled {done}/{len(todo)}")
            sys.stdout.flush()
        if args.sleep:
            time.sleep(args.sleep)

    actual = tok_in / 1e6 * in_price + tok_out / 1e6 * out_price
    print(f"\n\nLabelled {done} day(s), {failed} failure(s).")
    print(f"Tokens used: {tok_in:,} in / {tok_out:,} out")
    print(f"Actual cost: ~${actual:.4f}")
    print(f"Labels: {labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
