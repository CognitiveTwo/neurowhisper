"""Run the local steps of the toolkit in order and print where the report is.

By default this runs only the steps that never leave your machine
(``build_series.py``, optional ``calendar_features.py``, ``report.py``). Add
``--label`` and/or ``--cluster`` to include the two steps that call the OpenAI
API.

Examples::

    python analysis/run_all.py
    python analysis/run_all.py --ics ~/Downloads/mycalendar.ics
    python analysis/run_all.py --label --cluster
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import common

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *script_args):
    cmd = [sys.executable, os.path.join(HERE, script), *script_args]
    print(f"\n=== {script} {' '.join(script_args)} " + "=" * 20)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Run the analysis pipeline end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", action="store_true",
                   help="also run label_days.py (sends transcript text to OpenAI)")
    p.add_argument("--from-app-insights", action="store_true",
                   help="bootstrap labels from the app's ai_insights.json instead")
    p.add_argument("--cluster", action="store_true",
                   help="also run cluster_topics.py (sends topic phrases to OpenAI)")
    p.add_argument("--ics", metavar="FILE", help="calendar export to join in")
    p.add_argument("--model", default="gpt-4o-mini", help="chat model for the AI steps")
    p.add_argument("--title", default="Dictation report", help="report page title")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.from_app_insights:
        run("label_days.py", "--from-app-insights")
    if args.label:
        run("label_days.py", "--model", args.model)
    if args.cluster:
        run("cluster_topics.py", "--model", args.model)
    run("build_series.py")
    if args.ics:
        run("calendar_features.py", "--ics", args.ics)
    run("report.py", "--title", args.title)
    print(f"\nReport: {common.out_path('report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
