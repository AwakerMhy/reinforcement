"""Cohort-level paired analysis of the fixed CliffWalking margin selector."""

import csv
import json
from pathlib import Path

import numpy as np


def resample_interval(values, rng):
    bootstrap = np.empty(20000)
    for start in range(0, 20000, 1000):
        indices = rng.integers(0, len(values), (1000, len(values)))
        bootstrap[start:start + 1000] = values[indices].mean(axis=1)
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def describe(values, rng):
    return {"mean": float(values.mean()),
            "paired_se": float(values.std(ddof=1) / np.sqrt(len(values))),
            "bootstrap_95_percentile_interval": resample_interval(values, rng),
            "median": float(np.median(values)),
            "positive": int(np.sum(values > 0)),
            "negative": int(np.sum(values < 0)),
            "tie": int(np.sum(values == 0))}


def main():
    rows = list(csv.DictReader(Path("results/cliffwalking_margin.csv").open()))
    admitted = [r for r in rows if r["admitted"] == "1"]
    pos = np.array([int(r["positive_strong_steps"]) -
                    int(r["positive_weak_steps"]) for r in admitted])
    neg = np.array([int(r["negative_strong_steps"]) -
                    int(r["negative_weak_steps"]) for r in admitted])
    rng = np.random.default_rng(20261128)
    selection = sum(int(r["selection_assessment_steps"]) for r in rows)
    preparation = len(rows) * 64 * 2500
    result = {"cohorts": len(rows), "admitted": len(admitted),
              "reference_available": sum(bool(r["reference"]) for r in rows),
              "quality_classes_available": sum(
                  int(r["n_strong"]) > 0 and int(r["n_weak"]) > 0 for r in rows),
              "preparation_steps": preparation,
              "selection_assessment_steps": selection,
              "preparation_plus_selection_per_admitted":
                  (preparation + selection) / len(admitted),
              "continuation_assessment_steps": sum(
                  int(r["continuation_assessment_steps"]) for r in admitted),
              "positive_arm_gap": describe(pos, rng),
              "negative_arm_gap": describe(neg, rng),
              "paired_interaction": describe(pos - neg, rng),
              "censor_counts": {
                  key: sum(int(r[key]) for r in admitted)
                  for key in ("positive_strong_censored", "positive_weak_censored",
                              "negative_strong_censored", "negative_weak_censored")}}
    Path("results/cliffwalking_margin_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
