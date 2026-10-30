"""
Local TD-update checks.

AI assistance: OpenAI Codex.

Nonzero fixture weights are used only to test the arithmetic.
The actual solver must initialise all weights to zero.
"""

import math

from game_env import GameEnv, TrainingBudgetExhausted
from game_state import GameState
from solution import Solver


def dot(weights, features):
    return sum(
        weight * feature
        for weight, feature in zip(weights, features)
    )


def check_update(terminal):
    env = GameEnv("testcases/L1.txt")
    env.set_training_budget(1000, -1000000)
    env.max_episode_steps = 1

    solver = Solver(env)
    solver.EPSILON_START = 0.0
    solver.EPSILON_END = 0.0
    solver.fq_initialise()

    assert all(w == 0.0 for w in solver.fq_weights)

    # Artificial old weights are used only in this test.
    old = [
        0.01 * (i + 1)
        for i in range(len(solver.fq_weights))
    ]
    solver.fq_weights = list(old)

    state = env.get_init_state()

    status = tuple(
        1 if terminal else 0
        for _ in env.crystal_positions
    )

    next_state = GameState(
        *env.launch_positions[0], status
    )
    assert env.is_terminal(next_state) == terminal

    chosen = []

    def sample(current, action):
        assert current == state
        assert action in env.get_valid_actions(current)

        chosen.append(action)
        env.training_steps_used += 1

        return next_state, -2.0, None

    env.perform_action = sample
    solver.fq_train_episode()

    assert len(chosen) == 1

    features = solver.fq_get_features(
        state, chosen[0]
    )

    target = -2.0

    if not terminal:
        target += env.gamma * max(
            dot(
                old,
                solver.fq_get_features(next_state, action),
            )
            for action in env.get_valid_actions(next_state)
        )

    error = target - dot(old, features)

    expected = [
        weight + solver.fq_alpha * error * feature
        for weight, feature in zip(old, features)
    ]

    assert all(
        math.isclose(actual, wanted, abs_tol=1e-10)
        for actual, wanted in zip(
            solver.fq_weights, expected
        )
    )

    assert len(solver.fq_history) == 1
    assert solver.fq_history[0]["steps"] == 1

    print(
        f"PASS: exact FQ update, terminal={terminal}"
    )


check_update(True)
check_update(False)

# Test training in the real environment.
env = GameEnv("testcases/L1.txt")
env.set_training_budget(1000, -1000000)
env.max_episode_steps = 10

solver = Solver(env)
solver.fq_initialise()

solver.fq_train_episode()
solver.fq_train_episode()

assert len(solver.fq_history) == 2
assert env.training_steps_used > 0
assert any(w != 0.0 for w in solver.fq_weights)
assert all(math.isfinite(w) for w in solver.fq_weights)

# Evaluation must preserve the weights and training budget.
before = tuple(solver.fq_weights)
used = env.training_steps_used

state = env.get_init_state()
selected = solver.fq_select_action(state)

assert selected in env.get_valid_actions(state)
assert all(
    solver.fq_select_action(state) == selected
    for _ in range(10)
)

assert tuple(solver.fq_weights) == before
assert env.training_steps_used == used

print(
    "PASS: real FQ training, history "
    "and read-only greedy evaluation"
)

# The solver must let budget exhaustion propagate.
env.set_training_budget(0, -1000000)

try:
    solver.fq_train_episode()
except TrainingBudgetExhausted:
    print("PASS: training budget exception propagates")
else:
    raise AssertionError(
        "Training budget exception was swallowed"
    )