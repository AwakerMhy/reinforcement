"""Export the deterministic official Gymnasium CliffWalking transition table."""

import json
from pathlib import Path

import gymnasium as gym


def main():
    env = gym.make("CliffWalking-v1", is_slippery=False)
    table = []
    for state in range(env.observation_space.n):
        row = []
        for action in range(env.action_space.n):
            entries = env.unwrapped.P[state][action]
            if len(entries) != 1 or entries[0][0] != 1.0:
                raise ValueError("Expected a deterministic official transition")
            _, next_state, reward, terminal = entries[0]
            row.append([int(next_state), int(reward), bool(terminal)])
        table.append(row)
    env.close()
    payload = {"gymnasium_version": gym.__version__,
               "environment": "CliffWalking-v1", "is_slippery": False,
               "state_count": 48, "action_count": 4, "transitions": table}
    path = Path("experiments/cliffwalking_transitions.json")
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(path)


if __name__ == "__main__":
    main()
