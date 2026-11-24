"""Recompute the paired FrozenLake pilot statistics from the raw CSV."""

import csv
import json
from pathlib import Path

import numpy as np


def main():
    rows = [row for row in csv.DictReader(
        Path("results/frozenlake_pair_pilot.csv").open()) if row["admitted"] == "1"]
    strong = np.array([int(row["strong_hit"]) for row in rows])
    weak = np.array([int(row["weak_hit"]) for row in rows])
    sc = np.array([int(row["strong_censored"]) for row in rows], dtype=bool)
    wc = np.array([int(row["weak_censored"]) for row in rows], dtype=bool)
    gap = strong - weak
    bootstrap = np.random.default_rng(20261124).integers(0, len(rows), (20000, len(rows)))
    interval = np.quantile(gap[bootstrap].mean(axis=1), [0.025, 0.975])
    cdf = {str(t): {"strong": float(np.mean((strong <= t) & ~sc)),
                    "weak": float(np.mean((weak <= t) & ~wc))}
           for t in (500, 1000, 2000, 3000)}
    result = {"admitted": len(rows), "mean_gap_strong_minus_weak": float(gap.mean()),
              "paired_se": float(gap.std(ddof=1) / np.sqrt(len(rows))),
              "bootstrap_95_percentile_interval": interval.tolist(),
              "median_gap": float(np.median(gap)),
              "weak_first": int(np.sum(gap > 0)),
              "strong_first": int(np.sum(gap < 0)), "simultaneous": int(np.sum(gap == 0)),
              "strong_censored": int(sc.sum()), "weak_censored": int(wc.sum()),
              "hit_fraction_by_step": cdf}
    Path("results/frozenlake_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
