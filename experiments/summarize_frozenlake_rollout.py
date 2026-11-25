"""Analyze the fixed rollout-selected FrozenLake cohort CSV."""

import csv
import json
from pathlib import Path

import numpy as np


def describe(rows, seed):
    if not rows:
        return {"pairs": 0}
    strong = np.array([int(r["strong_hit"]) for r in rows])
    weak = np.array([int(r["weak_hit"]) for r in rows])
    sc = np.array([int(r["strong_censored"]) for r in rows], dtype=bool)
    wc = np.array([int(r["weak_censored"]) for r in rows], dtype=bool)
    gap = strong - weak
    indices = np.random.default_rng(seed).integers(0, len(rows), (20000, len(rows)))
    ci = np.quantile(gap[indices].mean(axis=1), [0.025, 0.975])
    return {"pairs": len(rows),
            "mean_strong": float(strong.mean()), "mean_weak": float(weak.mean()),
            "mean_gap_strong_minus_weak": float(gap.mean()),
            "paired_se": float(gap.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
            "bootstrap_95_percentile_interval": ci.tolist(),
            "median_gap": float(np.median(gap)),
            "weak_first": int(np.sum(gap > 0)), "strong_first": int(np.sum(gap < 0)),
            "simultaneous": int(np.sum(gap == 0)),
            "strong_censored": int(sc.sum()), "weak_censored": int(wc.sum()),
            "hit_fraction_by_step": {
                str(t): {"strong": float(np.mean((strong <= t) & ~sc)),
                         "weak": float(np.mean((weak <= t) & ~wc))}
                for t in (500, 1000, 2000, 3000)}}


def main():
    rows = list(csv.DictReader(Path("results/frozenlake_rollout_selection.csv").open()))
    selected = [r for r in rows if r["admitted"] == "1"]
    true_order = [r for r in selected if float(r["exact_initial_strong"]) >
                  float(r["exact_initial_weak"])]
    result = {"cohorts": len(rows), "admitted": len(selected),
              "true_initial_order": len(true_order),
              "preparation_steps": len(rows) * 32 * 1000,
              "selection_rollout_steps": sum(int(r["selection_env_steps"]) for r in rows),
              "all_admitted_primary": describe(selected, 20261125),
              "true_order_sensitivity": describe(true_order, 20261126)}
    Path("results/frozenlake_rollout_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
