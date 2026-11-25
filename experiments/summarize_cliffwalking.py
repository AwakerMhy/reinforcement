"""Recompute CliffWalking paired step statistics from the raw cohort CSV."""

import csv
import json
from pathlib import Path

import numpy as np


def main():
    rows = list(csv.DictReader(Path("results/cliffwalking_equal_steps.csv").open()))
    admitted = [row for row in rows if row["admitted"] == "1"]
    strong = np.array([int(row["strong_steps"]) for row in admitted])
    weak = np.array([int(row["weak_steps"]) for row in admitted])
    sc = np.array([int(row["strong_censored"]) for row in admitted], dtype=bool)
    wc = np.array([int(row["weak_censored"]) for row in admitted], dtype=bool)
    gap = strong - weak
    rng = np.random.default_rng(20261126)
    bootstrap = np.empty(20000)
    for start in range(0, 20000, 1000):
        indices = rng.integers(0, len(gap), (1000, len(gap)))
        bootstrap[start:start + 1000] = gap[indices].mean(axis=1)
    result = {"cohorts": len(rows), "admitted": len(admitted),
              "preparation_steps": len(rows) * 64 * 2500,
              "selection_assessment_steps": sum(int(row["selection_assessment_steps"])
                                                for row in rows),
              "continuation_assessment_steps": sum(
                  int(row["strong_assessment_steps"]) +
                  int(row["weak_assessment_steps"]) for row in admitted),
              "mean_strong_steps": float(strong.mean()),
              "mean_weak_steps": float(weak.mean()),
              "mean_gap_strong_minus_weak": float(gap.mean()),
              "paired_se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
              "bootstrap_95_percentile_interval": np.quantile(
                  bootstrap, [0.025, 0.975]).tolist(),
              "median_gap": float(np.median(gap)),
              "weak_first": int(np.sum(gap > 0)),
              "strong_first": int(np.sum(gap < 0)),
              "simultaneous": int(np.sum(gap == 0)),
              "strong_censored": int(sc.sum()), "weak_censored": int(wc.sum()),
              "hit_fraction_by_step": {
                  str(t): {"strong": float(np.mean((strong <= t) & ~sc)),
                           "weak": float(np.mean((weak <= t) & ~wc))}
                  for t in (25, 50, 100, 200, 500, 3000)}}
    Path("results/cliffwalking_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
