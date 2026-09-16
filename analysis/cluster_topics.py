"""Cluster the topic phrases from the day labels into named categories.

Steps:
  1. collect every unique topic phrase from ``analysis/out/labels/*.json``
  2. embed them with ``text-embedding-3-small`` (cached in ``out/embeddings.json``)
  3. k-means, with k chosen by silhouette score unless ``--k`` is given (local)
  4. a second k-means over the cluster centroids groups the clusters into 6-10
     categories (local), and one chat call names the clusters and categories

Only steps 2 and 4 leave your machine, and only short topic phrases are sent -
never raw transcript text.

Example::

    python analysis/cluster_topics.py
    python analysis/cluster_topics.py --k 12 --no-name
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import common

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_1M = 0.02  # USD, text-embedding-3-small
DEFAULT_NAME_MODEL = "gpt-4o-mini"
BATCH = 500

NAMING_PROMPT = """You are given clusters of short topic phrases taken from one
person's daily dictation notes. The clusters have already been grouped into
categories for you; each cluster carries the id of the group it belongs to.
Your only job is to NAME them.

Return ONLY valid JSON:
{
  "clusters": {"0": "short cluster name", "1": "..."},
  "categories": {"0": "Category name", "1": "..."},
  "definitions": {"Category name": "One sentence: what belongs here, with two short examples taken from the phrases."}
}

Rules:
- "clusters" must contain one name for every cluster id you were given.
- "categories" must contain one name for every group id you were given.
- Do not change the grouping, do not merge or split anything.
- A category name must fit every cluster in that group, so keep it a little
  broader than the cluster names.
- Names are 1-4 words, concrete, and drawn from the phrases themselves.
- Definitions are one sentence each and cite two short example phrases."""


def collect_phrases(labels):
    phrases = []  # (phrase, day)
    for day, value in sorted(labels.items()):
        for t in value.get("topics", []):
            t = str(t).strip()
            if t:
                phrases.append((t, day))
    return phrases


def embed_all(client, texts, cache, cache_path, verbose=True):
    todo = [t for t in texts if t not in cache]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for t, item in zip(chunk, resp.data):
            cache[t] = item.embedding
        if verbose:
            print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}")
        common.write_json(cache_path, cache, indent=None)
    if todo:
        common.write_json(cache_path, cache, indent=None)
    return cache


def choose_k(X, kmin, kmax, verbose=True):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = None
    for k in range(kmin, kmax + 1):
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        if verbose:
            print(f"  k={k:2d} silhouette={score:.3f}")
        if best is None or score > best[0]:
            best = (score, k)
    return best[1] if best else kmin


def group_clusters(centroids, k):
    """Group cluster centroids into 6-10 categories, locally and deterministically."""
    from sklearn.cluster import KMeans
    if k <= 6:
        return {c: c for c in range(k)}, k
    g = min(10, max(6, round(k / 2)))
    labels = KMeans(n_clusters=g, n_init=20, random_state=0).fit_predict(centroids)
    return {c: int(labels[c]) for c in range(k)}, g


def name_clusters(client, model, report, cluster_group, n_groups):
    payload = {
        "clusters": [{"cluster": r["cluster"], "group": cluster_group[r["cluster"]],
                      "n_mentions": r["n_mentions"],
                      "phrases": r["central"][:12] + r["frequent"][:8]}
                     for r in report],
        "group_ids": list(range(n_groups)),
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NAMING_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Embed and cluster topic phrases, then name the clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=0,
                   help="number of clusters (0 = pick by silhouette score)")
    p.add_argument("--k-min", type=int, default=8, help="lowest k to try")
    p.add_argument("--k-max", type=int, default=16, help="highest k to try")
    p.add_argument("--model", default=DEFAULT_NAME_MODEL,
                   help="chat model used to name clusters and categories")
    p.add_argument("--no-name", action="store_true",
                   help="skip the naming call; clusters keep numeric names")
    p.add_argument("--labels", metavar="DIR", help="labels directory")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="progress output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    labels = common.load_labels(args.labels)
    if not labels:
        raise SystemExit("No labels found. Run label_days.py first.")
    phrases = collect_phrases(labels)
    texts = sorted({t for t, _ in phrases})
    print(f"days {len(labels)}, topic mentions {len(phrases)}, unique phrases {len(texts)}")
    if len(texts) < 12:
        raise SystemExit("Too few unique topic phrases to cluster (need at least 12).")

    cache_path = common.out_path("embeddings.json")
    cache = common.read_json(cache_path, {}) or {}
    missing = [t for t in texts if t not in cache]
    est = sum(len(t) for t in missing) / 4 / 1e6 * EMBED_PRICE_PER_1M
    print(f"embeddings cached {len(texts) - len(missing)}, to fetch {len(missing)} "
          f"(~${est:.4f})")

    need_api = bool(missing) or not args.no_name
    client = None
    if need_api:
        key = common.require_api_key()
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("The openai package is missing: pip install -r analysis/requirements.txt")
        client = OpenAI(api_key=key)
    if missing:
        embed_all(client, texts, cache, cache_path, args.verbose)

    import numpy as np
    from sklearn.cluster import KMeans

    X = np.array([cache[t] for t in texts], dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    k = args.k or choose_k(X, args.k_min, min(args.k_max, len(texts) - 1), args.verbose)
    k = max(2, min(k, len(texts) - 1))
    lab = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)
    print(f"using k={k}")
    phrase_cluster = {t: int(c) for t, c in zip(texts, lab)}

    centroids = np.array([X[lab == c].mean(axis=0) for c in range(k)])
    mentions = collections.Counter(phrase_cluster[t] for t, _ in phrases)
    freq = collections.defaultdict(collections.Counter)
    for t, _ in phrases:
        freq[phrase_cluster[t]][t] += 1
    report = []
    for c in range(k):
        idx = [i for i in range(len(texts)) if lab[i] == c]
        sims = sorted(((float(X[i] @ centroids[c]), texts[i]) for i in idx), reverse=True)
        report.append({
            "cluster": c,
            "n_unique": len(idx),
            "n_mentions": mentions[c],
            "central": [t for _, t in sims[:12]],
            "frequent": [t for t, _ in freq[c].most_common(12)],
        })

    month_weights = collections.defaultdict(collections.Counter)
    for t, day in phrases:
        month_weights[day[:7]][phrase_cluster[t]] += 1

    cluster_group, n_groups = group_clusters(centroids, k)
    group_names = {}
    names, definitions = {}, {}
    if not args.no_name:
        print(f"naming {k} clusters and {n_groups} categories ...")
        try:
            named = name_clusters(client, args.model, report, cluster_group, n_groups)
            names = {str(kk): str(v).strip() for kk, v
                     in (named.get("clusters") or {}).items() if str(v).strip()}
            group_names = {str(kk): str(v).strip() for kk, v
                           in (named.get("categories") or {}).items() if str(v).strip()}
            definitions = {str(kk): str(v) for kk, v
                           in (named.get("definitions") or {}).items()}
        except Exception as exc:
            print(f"  naming failed ({exc}); falling back to numeric names")
    for c in range(k):
        names.setdefault(str(c), f"Cluster {c}")
    for g in range(n_groups):
        group_names.setdefault(str(g), f"Group {g}")
    # A name collision would silently merge two categories - keep them distinct.
    seen = {}
    for g in range(n_groups):
        base = group_names[str(g)]
        if base in seen:
            group_names[str(g)] = f"{base} ({g})"
        seen[group_names[str(g)]] = g

    cluster_to_category = {str(c): group_names[str(cluster_group[c])] for c in range(k)}
    weight = collections.Counter()
    for r in report:
        weight[cluster_to_category[str(r["cluster"])]] += r["n_mentions"]
    order = [cat for cat, _ in weight.most_common()]

    for r in report:
        r["name"] = names[str(r["cluster"])]
        r["category"] = cluster_to_category[str(r["cluster"])]

    common.write_json(common.out_path("clusters.json"), {
        "k": k,
        "clusters": report,
        "phrase_cluster": phrase_cluster,
        "month_weights": {m: dict(v) for m, v in sorted(month_weights.items())},
    })
    common.write_json(common.out_path("categories.json"), {
        "cluster_names": names,
        "cluster_to_category": cluster_to_category,
        "category_order": order,
        "definitions": definitions,
    })

    lines = [f"days {len(labels)}, phrases {len(phrases)}, unique {len(texts)}",
             f"k = {k}", ""]
    for r in sorted(report, key=lambda r: -r["n_mentions"]):
        lines.append(f"== cluster {r['cluster']}: {r['name']}  [{r['category']}]  "
                     f"mentions={r['n_mentions']} unique={r['n_unique']}")
        lines.append("   central : " + "; ".join(r["central"][:8]))
        lines.append("   frequent: " + "; ".join(r["frequent"][:8]))
        lines.append("")
    lines.append("month | " + " ".join(f"c{c:<3d}" for c in range(k)))
    for m in sorted(month_weights):
        tot = sum(month_weights[m].values()) or 1
        lines.append(m + " " + " ".join(
            f"{int(100 * month_weights[m][c] / tot):4d}" for c in range(k)))
    report_txt = "\n".join(lines) + "\n"
    with open(common.out_path("cluster_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_txt)

    print("\n" + report_txt)
    print(f"Wrote {os.path.join(common.OUT_DIR, 'clusters.json')}, categories.json, "
          f"cluster_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
