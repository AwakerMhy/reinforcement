"""Equal-step paired Q-learning pilot on official slippery FrozenLake-v1.

This is an exploratory simulator study. Candidate selection uses the exact
Gymnasium transition model, so it is an oracle selection rule, not a deployable
way to choose learners on an unknown task.
"""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


def make_model():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    model = env.unwrapped.P
    assert env.spec.max_episode_steps == 100
    env.close()
    return model


def value(q, model):
    policy = np.argmax(q, axis=1)
    transition = np.zeros((16, 16))
    goal = np.zeros(16)
    for state, action in enumerate(policy):
        for probability, next_state, reward, terminal in model[state][int(action)]:
            goal[state] += probability * reward
            if not terminal:
                transition[state, next_state] += probability
    v = np.zeros(16)
    for _ in range(100):
        v = goal + transition @ v
    return float(v[0])


class Learner:
    def __init__(self, q):
        self.q = q
        self.state = 0
        self.episode_steps = 0

    def step(self, model, uniforms):
        explore, random_action, transition_u = uniforms
        action = (min(3, int(random_action * 4)) if explore < 0.2
                  else int(np.argmax(self.q[self.state])))
        choices = model[self.state][action]
        cumulative = 0.0
        for probability, next_state, reward, terminal in choices:
            cumulative += probability
            if transition_u < cumulative:
                break
        old_state = self.state
        target = reward + (0 if terminal else 0.99 * np.max(self.q[next_state]))
        self.q[old_state, action] += 0.5 * (target - self.q[old_state, action])
        self.episode_steps += 1
        if terminal or self.episode_steps == 100:
            self.state = 0
            self.episode_steps = 0
        else:
            self.state = int(next_state)


def candidate(model, seed, prep_steps):
    rng = np.random.default_rng(seed)
    learner = Learner(rng.normal(0, 0.001, (16, 4)))
    for _ in range(prep_steps):
        learner.step(model, rng.random(3))
    return learner


def run(args):
    model = make_model()
    rows = []
    for cohort in range(args.cohorts):
        pool = [candidate(model, np.random.SeedSequence([args.seed, cohort, j]),
                          args.prep_steps) for j in range(args.pool)]
        qualities = np.asarray([value(x.q, model) for x in pool])
        order = np.argsort(-qualities, kind="stable")
        strong, weak = pool[order[args.pool // 4 - 1]], pool[order[3 * args.pool // 4 - 1]]
        vs, vw = qualities[order[args.pool // 4 - 1]], qualities[order[3 * args.pool // 4 - 1]]
        admitted = bool(vs > vw + args.min_gap and vs < args.target and vw < args.target)
        record = {"cohort": cohort, "candidate_count": args.pool,
                  "prep_steps_each": args.prep_steps, "initial_strong": vs,
                  "initial_weak": vw, "admitted": int(admitted),
                  "strong_hit": "", "weak_hit": "", "strong_censored": "",
                  "weak_censored": ""}
        if admitted:
            strong.state = weak.state = 0
            strong.episode_steps = weak.episode_steps = 0
            future_rng = np.random.default_rng(np.random.SeedSequence([args.seed, cohort, 999999]))
            ts = tw = None
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
            record.update(strong_hit=ts or args.continuation_cap,
                          weak_hit=tw or args.continuation_cap,
                          strong_censored=int(ts is None),
                          weak_censored=int(tw is None))
        rows.append(record)
    accepted = [r for r in rows if r["admitted"]]
    gap = np.asarray([r["strong_hit"] - r["weak_hit"] for r in accepted], dtype=float)
    summary = {"experiment": "exploratory_equal_step_frozenlake_pair",
               "gymnasium_version": gym.__version__, "environment": "FrozenLake-v1",
               "map": "4x4", "is_slippery": True, "selection": "oracle exact V, ranks pool/4 and 3pool/4",
               "parameters": vars(args), "cohorts": len(rows), "admitted": len(accepted),
               "mean_initial_strong": float(np.mean([r["initial_strong"] for r in accepted])) if accepted else None,
               "mean_initial_weak": float(np.mean([r["initial_weak"] for r in accepted])) if accepted else None,
               "mean_capped_strong_hit": float(np.mean([r["strong_hit"] for r in accepted])) if accepted else None,
               "mean_capped_weak_hit": float(np.mean([r["weak_hit"] for r in accepted])) if accepted else None,
               "mean_gap_strong_minus_weak": float(gap.mean()) if len(gap) else None,
               "gap_se": float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else None,
               "weak_first_fraction": float(np.mean(gap > 0)) if len(gap) else None,
               "strong_censored": sum(r["strong_censored"] for r in accepted),
               "weak_censored": sum(r["weak_censored"] for r in accepted)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=int, default=100)
    parser.add_argument("--pool", type=int, default=32)
    parser.add_argument("--prep-steps", type=int, default=1000)
    parser.add_argument("--continuation-cap", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", default="results/frozenlake_pair_pilot")
    args = parser.parse_args()
    if args.pool < 4 or args.pool % 4 or args.continuation_cap % args.eval_interval:
        parser.error("pool must be divisible by 4 and cap by eval interval")
    return args


if __name__ == "__main__":
    run(parse_args())
