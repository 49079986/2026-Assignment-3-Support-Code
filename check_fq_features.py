"""
Local feature and greedy-selection checks.

AI assistance: OpenAI Codex.
"""

import math

from game_env import GameEnv
from game_state import GameState
from harness import check_feature_vectors
from solution import Solver


for level in ("L1", "L2", "L3", "L4"):
    env = GameEnv(f"testcases/{level}.txt")
    env.set_training_budget(1000, -100000)

    solver = Solver(env)
    solver.fq_initialise()

    assert env.training_steps_used == 0
    assert len(solver.fq_weights) == 21
    assert len(solver.FEATURE_NAMES) == 21
    assert all(
        weight == 0.0 for weight in solver.fq_weights
    )

    length, problem = check_feature_vectors(
        env, solver.fq_get_features
    )

    assert problem is None, problem
    assert length == 21

    no_crystals = (0,) * len(env.crystal_positions)

    states = [env.get_init_state()]
    states += [
        GameState(row, col, no_crystals)
        for row, col in env.generalisation_start_positions
    ]

    ready_state = GameState(
        env.init_row,
        env.init_col,
        (1,) * len(env.crystal_positions),
    )
    states.append(ready_state)

    for state in states:
        for action in env.get_valid_actions(state):
            features = solver.fq_get_features(state, action)

            assert len(features) == 21
            assert all(
                math.isfinite(value) for value in features
            )
            assert (
                sum(value * value for value in features)
                <= 1.0 + 1e-10
            )

    start = env.get_init_state()
    action = env.get_valid_actions(start)[0]

    ready_index = solver.FEATURE_NAMES.index(
        "ready_for_launch"
    )

    assert (
        solver.fq_get_features(
            ready_state, action
        )[ready_index] > 0.0
    )

    if env.min_samples > 0:
        assert (
            solver.fq_get_features(
                start, action
            )[ready_index] == 0.0
        )

    # Test-only weight checks that selection uses the features.
    right_index = solver.FEATURE_NAMES.index("right")
    solver.fq_weights[right_index] = 1.0

    before = tuple(solver.fq_weights)
    selected = solver.fq_select_action(start)

    assert selected in env.get_valid_actions(start)
    assert selected.endswith("r")

    assert all(
        solver.fq_select_action(start) == selected
        for _ in range(10)
    )

    assert tuple(solver.fq_weights) == before
    assert env.training_steps_used == 0

    print(
        f"PASS {level}: zero initial weights, "
        "21 finite normalised features, target switching "
        "and deterministic greedy selection"
    )

print("All feature checks passed.")