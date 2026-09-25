from pathlib import Path
import zipfile
import io
import gc
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path("external_data/s6e9_zoom_zoom_baseline")
ARCHIVE = ROOT / "ranked_predictions_latest.zip.bin"
CATALOG = ROOT / "ranked_catalog_latest.csv"
BASELINE = Path("submissions/submission_14.csv")

print("=" * 100)
print("EXP45: EXTERNAL PREDICTION FAMILY ANALYSIS")
print("=" * 100)

catalog = pd.read_csv(CATALOG)
baseline = pd.read_csv(BASELINE)

base_ids = baseline["id"].to_numpy()
base_pred = baseline["Will_Buy_EV"].to_numpy(dtype=np.float64)

base_rank = pd.Series(base_pred).rank(method="average").to_numpy(dtype=np.float64)

top_catalog = (
    catalog[
        catalog["public_score"].notna()
        & (catalog["rows"] == len(baseline))
        & (catalog["public_score"] >= 0.94630)
    ]
    .sort_values(
        ["public_score", "display_order"],
        ascending=[False, True]
    )
    .head(260)
    .copy()
)

print("Candidates loaded from catalog:", len(top_catalog))
print(
    "Public score range:",
    top_catalog["public_score"].min(),
    "to",
    top_catalog["public_score"].max()
)

predictions = {}
metadata = {}

with zipfile.ZipFile(ARCHIVE, "r") as z:

    for n, row in enumerate(
        top_catalog.itertuples(index=False),
        start=1
    ):
        member = row.csv_path

        try:
            raw = z.read(member)
            df = pd.read_csv(io.BytesIO(raw))

            if not {"id", "Will_Buy_EV"}.issubset(df.columns):
                continue

            if len(df) != len(baseline):
                continue

            ids = df["id"].to_numpy()

            if not np.array_equal(ids, base_ids):
                continue

            pred = df["Will_Buy_EV"].to_numpy(dtype=np.float64)

            predictions[member] = pred

            metadata[member] = {
                "public_score": float(row.public_score),
                "origin": row.origin,
                "author": row.author,
                "submission_id": row.submission_id,
                "version_id": row.version_id,
            }

        except Exception as e:
            print("ERROR:", member, "|", repr(e))

        if n % 25 == 0:
            print(f"Loaded {n}/{len(top_catalog)}")

        gc.collect()

print("\nCompatible predictions:", len(predictions))

if len(predictions) < 10:
    raise RuntimeError("Too few compatible predictions for Exp45.")

# -------------------------------------------------------------------------
# 1. Pairwise diversity
# -------------------------------------------------------------------------

members = list(predictions.keys())
rank_cache = {}

for member in members:
    rank_cache[member] = (
        pd.Series(predictions[member])
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )

rows = []

for a, b in itertools.combinations(members, 2):

    rank_a = rank_cache[a]
    rank_b = rank_cache[b]

    rho = spearmanr(rank_a, rank_b).statistic
    rank_diff = np.abs(rank_a - rank_b)

    rows.append({
        "member_a": a,
        "member_b": b,
        "score_a": metadata[a]["public_score"],
        "score_b": metadata[b]["public_score"],
        "author_a": metadata[a]["author"],
        "author_b": metadata[b]["author"],
        "submission_a": metadata[a]["submission_id"],
        "submission_b": metadata[b]["submission_id"],
        "spearman": float(rho),
        "mean_abs_rank_diff": float(rank_diff.mean()),
        "median_abs_rank_diff": float(np.median(rank_diff)),
        "p95_abs_rank_diff": float(np.percentile(rank_diff, 95)),
        "p99_abs_rank_diff": float(np.percentile(rank_diff, 99)),
    })

pairwise = pd.DataFrame(rows)

print("\n" + "=" * 100)
print("MOST SIMILAR HIGH-SCORING PREDICTIONS")
print("=" * 100)

print(
    pairwise
    .sort_values("spearman", ascending=False)
    [
        [
            "score_a",
            "score_b",
            "author_a",
            "author_b",
            "submission_a",
            "submission_b",
            "spearman",
            "mean_abs_rank_diff"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 100)
print("MOST DIFFERENT HIGH-SCORING PAIRS")
print("=" * 100)

print(
    pairwise
    .sort_values(
        ["spearman", "score_a", "score_b"],
        ascending=[True, False, False]
    )
    [
        [
            "score_a",
            "score_b",
            "author_a",
            "author_b",
            "submission_a",
            "submission_b",
            "spearman",
            "mean_abs_rank_diff",
            "median_abs_rank_diff",
            "p99_abs_rank_diff"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

# -------------------------------------------------------------------------
# 2. Compare every external prediction against Submission 14
# -------------------------------------------------------------------------

diversity_rows = []

for member in members:

    rank = rank_cache[member]

    rho = spearmanr(base_rank, rank).statistic
    rank_diff = np.abs(base_rank - rank)

    diversity_rows.append({
        "csv_path": member,
        "public_score": metadata[member]["public_score"],
        "origin": metadata[member]["origin"],
        "author": metadata[member]["author"],
        "submission_id": metadata[member]["submission_id"],
        "spearman_to_s14": float(rho),
        "mean_abs_rank_diff": float(rank_diff.mean()),
        "median_abs_rank_diff": float(np.median(rank_diff)),
        "p95_abs_rank_diff": float(np.percentile(rank_diff, 95)),
        "p99_abs_rank_diff": float(np.percentile(rank_diff, 99)),
    })

diversity = pd.DataFrame(diversity_rows)

print("\n" + "=" * 100)
print("BEST EXTERNAL SCORE WITH LOWEST CORRELATION TO S14")
print("=" * 100)

print(
    diversity
    .sort_values(
        ["spearman_to_s14", "public_score"],
        ascending=[True, False]
    )
    [
        [
            "public_score",
            "origin",
            "author",
            "submission_id",
            "spearman_to_s14",
            "mean_abs_rank_diff",
            "median_abs_rank_diff",
            "p99_abs_rank_diff"
        ]
    ]
    .head(40)
    .to_string(index=False)
)

# -------------------------------------------------------------------------
# 3. Author-level summary
# -------------------------------------------------------------------------

print("\n" + "=" * 100)
print("AUTHOR SUMMARY")
print("=" * 100)

author_summary = (
    diversity
    .groupby("author")
    .agg(
        files=("submission_id", "count"),
        best_public_score=("public_score", "max"),
        mean_public_score=("public_score", "mean"),
        min_spearman_to_s14=("spearman_to_s14", "min"),
        mean_spearman_to_s14=("spearman_to_s14", "mean"),
        mean_rank_diff=("mean_abs_rank_diff", "mean"),
    )
    .sort_values(
        ["best_public_score", "min_spearman_to_s14"],
        ascending=[False, True]
    )
)

print(author_summary.head(40).to_string())

# -------------------------------------------------------------------------
# 4. Elite family analysis
# -------------------------------------------------------------------------

elite = diversity[
    diversity["public_score"] >= 0.94650
].copy()

print("\n" + "=" * 100)
print("ELITE FAMILY")
print("=" * 100)

print("Elite files >= 0.94650:", len(elite))

if len(elite) >= 2:

    elite_members = elite["csv_path"].tolist()

    elite_corr = []

    for a, b in itertools.combinations(elite_members, 2):

        elite_corr.append({
            "submission_a": metadata[a]["submission_id"],
            "submission_b": metadata[b]["submission_id"],
            "score_a": metadata[a]["public_score"],
            "score_b": metadata[b]["public_score"],
            "spearman": float(
                spearmanr(
                    rank_cache[a],
                    rank_cache[b]
                ).statistic
            ),
        })

    elite_corr_df = pd.DataFrame(elite_corr)

    print(
        elite_corr_df
        .sort_values("spearman")
        .head(30)
        .to_string(index=False)
    )

    print("\nElite pairwise Spearman:")
    print(
        "min =",
        elite_corr_df["spearman"].min(),
        "| median =",
        elite_corr_df["spearman"].median(),
        "| max =",
        elite_corr_df["spearman"].max()
    )

# -------------------------------------------------------------------------
# 5. Score bands
# -------------------------------------------------------------------------

print("\n" + "=" * 100)
print("SCORE BAND DIVERSITY")
print("=" * 100)

for threshold in [0.94650, 0.94640, 0.94630]:

    subset = diversity[
        diversity["public_score"] >= threshold
    ]

    print(
        f">= {threshold:.5f}:",
        len(subset),
        "files |",
        "min S14 Spearman:",
        f"{subset['spearman_to_s14'].min():.6f}",
        "| mean:",
        f"{subset['spearman_to_s14'].mean():.6f}"
    )

# -------------------------------------------------------------------------
# 6. Save compact analysis outputs
# -------------------------------------------------------------------------

pairwise_output = ROOT / "exp45_external_pairwise.csv"
diversity_output = ROOT / "exp45_external_vs_s14.csv"
author_output = ROOT / "exp45_author_summary.csv"

pairwise.to_csv(pairwise_output, index=False)
diversity.to_csv(diversity_output, index=False)
author_summary.reset_index().to_csv(author_output, index=False)

print("\nSaved:")
print(pairwise_output)
print(diversity_output)
print(author_output)

print("\n" + "=" * 100)
print("EXP45 COMPLETE")
print("=" * 100)
