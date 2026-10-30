import math

from game_env import GameEnv, TrainingBudgetExhausted
from game_state import GameState
from solution import Solver


def check_update(terminal):
    env = GameEnv("testcases/L1.txt")
    env.set_training_budget(1000, -100000)
    env.max_episode_steps = 1

    solver = Solver(env)
    solver.EPSILON_START = 0.0
    solver.EPSILON_END = 0.0
    solver.ql_initialise()

    state = env.get_init_state()
    action = env.get_valid_actions(state)[0]
    index = env.ACTIONS.index(action)

    crystal_status = (
        (1,) * len(env.crystal_positions)
        if terminal
        else (0,) * len(env.crystal_positions)
    )

    next_state = GameState(
        *env.launch_positions[0], crystal_status
    )
    assert env.is_terminal(next_state) == terminal

    solver.q_table[state] = [0.0] * len(env.ACTIONS)
    solver.q_table[state][index] = 1.0

    # A terminal state's large stored values must be ignored.
    solver.q_table[next_state] = [
        10000.0 if terminal else 5.0
    ] * len(env.ACTIONS)

    def sample(current, selected_action):
        assert current == state
        assert selected_action == action
        env.training_steps_used += 1
        return next_state, -2.0, None

    env.perform_action = sample
    solver.ql_train_episode()

    target = -2.0
    if not terminal:
        target += env.gamma * 5.0

    expected = 1.0 + 0.2 * (target - 1.0)

    assert math.isclose(
        solver.q_table[state][index],
        expected,
        abs_tol=1e-10,
    )

    assert len(solver.ql_history) == 1
    assert solver.ql_history[0]["steps"] == 1

    print(
        f"PASS: terminal={terminal}, "
        f"updated Q={expected:.6f}"
    )


check_update(terminal=True)
check_update(terminal=False)

# Check two episodes in the real environment.
env = GameEnv("testcases/L1.txt")
env.set_training_budget(1000, -100000)
env.max_episode_steps = 10

solver = Solver(env)
solver.ql_initialise()

solver.ql_train_episode()
solver.ql_train_episode()

assert len(solver.ql_history) == 2
assert env.training_steps_used > 0
assert all(
    math.isfinite(value)
    for row in solver.q_table.values()
    for value in row
)
print("PASS: real episodes update Q-table and record history")

# Budget exceptions must propagate out of the solver.
env.set_training_budget(0, -100000)

try:
    solver.ql_train_episode()
except TrainingBudgetExhausted:
    print("PASS: training budget exception propagates")
else:
    raise AssertionError("Training budget exception was swallowed")