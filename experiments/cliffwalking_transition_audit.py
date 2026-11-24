"""Compare the previously used deterministic CliffWalking rule with Gymnasium."""

import json
from collections import deque

import gymnasium as gym


def previous_move(state: int, action: int) -> tuple[int, int, bool]:
    row, col = divmod(state, 12)
    if action == 0:
        row = max(0, row - 1)
    elif action == 1:
        col = min(11, col + 1)
    elif action == 2:
        row = min(3, row + 1)
    elif action == 3:
        col = max(0, col - 1)
    if row == 3 and 1 <= col <= 10:
        return 36, -100, False
    next_state = row * 12 + col
    return next_state, -1, next_state == 47


def main() -> None:
    env = gym.make("CliffWalking-v1", is_slippery=False)
    transitions = env.unwrapped.P
    queue = deque([36])
    reachable = {36}
    mismatches = []
    checked = 0
    while queue:
        state = queue.popleft()
        for action in range(4):
            entries = transitions[state][action]
            assert len(entries) == 1 and entries[0][0] == 1.0
            _, next_state, reward, terminal = entries[0]
            official = (int(next_state), int(reward), bool(terminal))
            prior = previous_move(state, action)
            checked += 1
            if official != prior:
                mismatches.append({"state": state, "action": action,
                                   "official": official, "prior": prior})
            if not terminal and int(next_state) not in reachable:
                reachable.add(int(next_state))
                queue.append(int(next_state))
    env.close()
    print(json.dumps({"gymnasium_version": gym.__version__, "environment":
                      "CliffWalking-v1", "is_slippery": False,
                      "reachable_nonterminal_states": len(reachable),
                      "checked_state_actions": checked,
                      "mismatches": mismatches}, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
