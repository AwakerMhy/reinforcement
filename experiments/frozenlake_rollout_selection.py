"""FrozenLake pair selection using official rollout outcomes, not exact values.

Preparation and continuation sample Gymnasium's official transition table.
Candidate selection calls the Gymnasium environment with held-out seeds.
The exact transition table is used only for post-selection outcome measurement.
"""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from frozenlake_pair_pilot import candidate, make_model, value


def rollout_score(env, q, seeds):
    wins = steps = 0
    for seed in seeds:
        state, _ = env.reset(seed=int(seed))
        for _ in range(100):
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            steps += 1
            if terminated or truncated:
                wins += int(reward > 0)
                break
        else:
            raise RuntimeError("Gymnasium's 100-step limit did not truncate")
    return wins / len(seeds), steps


def run(args):
    model = make_model()
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    rows = []
    for cohort in range(args.cohorts):
        pool = [candidate(model, np.random.SeedSequence([args.seed, cohort, j]),
                          args.prep_steps) for j in range(args.pool)]
        eval_seeds = [args.seed + cohort * 1000 + episode
                      for episode in range(args.selection_episodes)]
        scored = [rollout_score(env, learner.q, eval_seeds) for learner in pool]
        scores = np.array([s for s, _ in scored])
        evaluation_steps = sum(steps for _, steps in scored)
        order = np.argsort(-scores, kind="stable")
        si, wi = int(order[args.pool // 4 - 1]), int(order[3 * args.pool // 4 - 1])
        strong, weak = pool[si], pool[wi]
        selected_s, selected_w = float(scores[si]), float(scores[wi])
        admitted = bool(selected_s > selected_w + args.min_gap and
                        selected_s < args.target and selected_w < args.target)
        # These values are withheld from selection and used for audit/outcomes only.
        true_s, true_w = value(strong.q, model), value(weak.q, model)
        row = {"cohort": cohort, "admitted": int(admitted),
               "strong_candidate": si, "weak_candidate": wi,
               "selection_strong": selected_s, "selection_weak": selected_w,
               "exact_initial_strong": true_s, "exact_initial_weak": true_w,
               "selection_env_steps": evaluation_steps,
               "strong_hit": "", "weak_hit": "",
               "strong_censored": "", "weak_censored": ""}
        if admitted:
            strong.state = weak.state = 0
            strong.episode_steps = weak.episode_steps = 0
            future_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, cohort, 999999]))
            ts = 0 if true_s >= args.target else None
            tw = 0 if true_w >= args.target else None
            for step in range(args.eval_interval, args.continuation_cap + 1,
                              args.eval_interval):
                for _ in range(args.eval_interval):
                    uniforms = future_rng.random(3)
                    strong.step(model, uniforms)
                    weak.step(model, uniforms)
                if ts is None and value(strong.q, model) >= args.target:
                    ts = step
                if tw is None and value(weak.q, model) >= args.target:
                    tw = step
                if ts is not None and tw is not None:
                    break
            row.update(strong_hit=ts if ts is not None else args.continuation_cap,
                       weak_hit=tw if tw is not None else args.continuation_cap,
                       strong_censored=int(ts is None),
                       weak_censored=int(tw is None))
        rows.append(row)
    env.close()
    accepted = [r for r in rows if r["admitted"]]
    gap = np.array([r["strong_hit"] - r["weak_hit"] for r in accepted])
    summary = {"experiment": "rollout_selected_equal_step_frozenlake",
               "gymnasium_version": gym.__version__, "environment": "FrozenLake-v1",
               "map": "4x4", "is_slippery": True, "parameters": vars(args),
               "cohorts": len(rows), "admitted": len(accepted),
               "exact_initial_order_fraction": float(np.mean([
                   r["exact_initial_strong"] > r["exact_initial_weak"]
                   for r in accepted])) if accepted else None,
               "exact_initial_mean_strong": float(np.mean([
                   r["exact_initial_strong"] for r in accepted])) if accepted else None,
               "exact_initial_mean_weak": float(np.mean([
                   r["exact_initial_weak"] for r in accepted])) if accepted else None,
               "baseline_target_reached_strong": sum(
                   r["exact_initial_strong"] >= args.target for r in accepted),
               "baseline_target_reached_weak": sum(
                   r["exact_initial_weak"] >= args.target for r in accepted),
               "mean_capped_hit_strong": float(np.mean([
                   r["strong_hit"] for r in accepted])) if accepted else None,
               "mean_capped_hit_weak": float(np.mean([
                   r["weak_hit"] for r in accepted])) if accepted else None,
               "mean_gap_strong_minus_weak": float(gap.mean()) if len(gap) else None,
               "gap_se": float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else None,
               "weak_first_fraction": float(np.mean(gap > 0)) if len(gap) else None,
               "strong_censored": sum(r["strong_censored"] for r in accepted),
               "weak_censored": sum(r["weak_censored"] for r in accepted),
               "total_preparation_steps": len(rows) * args.pool * args.prep_steps,
               "total_selection_rollout_steps": sum(r["selection_env_steps"] for r in rows)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=int, default=100)
    parser.add_argument("--pool", type=int, default=32)
    parser.add_argument("--prep-steps", type=int, default=1000)
    parser.add_argument("--selection-episodes", type=int, default=200)
    parser.add_argument("--continuation-cap", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", default="results/frozenlake_rollout_selection")
    args = parser.parse_args()
    if (args.cohorts < 1 or args.pool < 4 or args.pool % 4 or
            args.continuation_cap % args.eval_interval or
            args.selection_episodes < 1):
        parser.error("invalid cohort, pool, episode, or evaluation interval")
    return args


if __name__ == "__main__":
    run(parse_args())
