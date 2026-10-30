"""
A3 official-budget experiments.

AI assistance: OpenAI Codex.
"""

from bisect import bisect_left
from datetime import datetime
import hashlib
import json
from pathlib import Path

from run_ql_rates import moving_average, write_csv
import matplotlib.pyplot as plt

from game_env import GameEnv, TrainingBudgetExhausted
from harness import (
    EPISODE_TRIALS,
    check_feature_vectors,
    check_initial_weights,
    evaluate_final_policy,
    evaluate_generalisation,
    get_learning_threshold,
    get_phase_value,
    seed_solver_randomness,
    train_agent,
)
from solution import Solver


ROOT = Path(__file__).resolve().parent
LEVELS = ("L1", "L2", "L3", "L4")


def run_one(level, algorithm, output):
    testcase = str(ROOT / "testcases" / f"{level}.txt")
    training_env = GameEnv(testcase)
    control_env = GameEnv(testcase)

    step_budget = get_phase_value(
        training_env, algorithm, "training_step_budget"
    )
    reward_budget = get_phase_value(
        training_env, algorithm, "training_reward_budget"
    )

    training_env.set_training_budget(
        step_budget, reward_budget
    )
    seed_solver_randomness(control_env.episode_seed)

    solver = Solver(training_env)

    result = train_agent(
        training_env,
        control_env,
        solver,
        algorithm,
        TrainingBudgetExhausted,
    )

    select_action = getattr(
        solver, f"{algorithm}_select_action"
    )

    feature_count = None
    weights = None

    if algorithm == "fq":
        feature_count, problem = check_feature_vectors(
            control_env, solver.fq_get_features
        )

        if problem is not None:
            raise ValueError(problem)

        check_initial_weights(
            solver,
            result.initial_numeric_arrays,
            feature_count,
        )

        assert (
            feature_count
            == len(solver.FEATURE_NAMES)
            == len(solver.fq_weights)
        )

        weights = list(solver.fq_weights)

        write_csv(
            output / f"{level}_fq_weights.csv",
            [
                {"feature": name, "weight": weight}
                for name, weight in zip(
                    solver.FEATURE_NAMES, weights
                )
            ],
        )

    mean_reward, solved = evaluate_final_policy(
        control_env, select_action, algorithm
    )

    generalisation = (
        evaluate_generalisation(control_env, select_action)
        if algorithm == "fq"
        else None
    )

    history = getattr(solver, f"{algorithm}_history")
    means = moving_average(
        [row["reward"] for row in history]
    )

    history = [
        dict(row, moving_average_50=mean)
        for row, mean in zip(history, means)
    ]

    if not history:
        raise RuntimeError(
            "No training episodes were recorded"
        )

    write_csv(
        output / f"{level}_{algorithm}_episodes.csv",
        history,
    )

    episode_steps = [
        row["training_steps"] for row in history
    ]

    checkpoints = [
        {
            "training_steps": steps,
            "training_episode": (
                bisect_left(episode_steps, steps) + 1
            ),
            "mean_greedy_reward": reward,
        }
        for steps, reward in result.checkpoint_history
    ]

    if checkpoints:
        write_csv(
            output / f"{level}_{algorithm}_checkpoints.csv",
            checkpoints,
        )

    learned_episode = None

    if result.learned_at_steps is not None:
        learned_episode = bisect_left(
            episode_steps, result.learned_at_steps
        ) + 1

    summary = {
        "level": level,
        "algorithm": algorithm,
        "protocol": "official_budget",
        "alpha": getattr(solver, f"{algorithm}_alpha"),
        "seed": control_env.episode_seed,
        "gamma": control_env.gamma,
        "max_episode_steps": control_env.max_episode_steps,
        "epsilon_start": solver.EPSILON_START,
        "epsilon_end": solver.EPSILON_END,
        "epsilon_decay_fraction": solver.EPSILON_DECAY_FRACTION,
        "step_budget": step_budget,
        "reward_budget": reward_budget,
        "training_steps": result.steps_used,
        "training_reward": result.training_reward,
        "training_episodes": len(history),
        "harness_episode_calls": result.episodes,
        "learning_threshold": get_learning_threshold(
            control_env, algorithm
        ),
        "learned_at_steps": result.learned_at_steps,
        "learned_at_episodes": learned_episode,
        "evaluation_reward_floor": get_phase_value(
            control_env, algorithm, "reward_min_tgt"
        ),
        "evaluation_mean_reward": mean_reward,
        "solved_trials": solved,
        "evaluation_trials": EPISODE_TRIALS,
        "generalisation_mean_reward": generalisation,
        "feature_count": feature_count,
        "time_per_training_step": result.get_time_per_step(),
        "solver_sha256": hashlib.sha256(
            (ROOT / "solution.py").read_bytes()
        ).hexdigest(),
    }

    print(
        f"{level} {algorithm.upper()}: "
        f"{result.steps_used} steps, "
        f"mean reward={mean_reward:.2f}, "
        f"solved={solved}/{EPISODE_TRIALS}",
        flush=True,
    )

    return history, summary, weights


def main():
    output = ROOT / "results" / (
        "baseline_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    runs = {}
    summaries = []
    weight_vectors = {}

    for level in LEVELS:
        for algorithm in ("ql", "fq"):
            history, summary, weights = run_one(
                level, algorithm, output
            )

            runs[(level, algorithm)] = history
            summaries.append(summary)

            if weights is not None:
                weight_vectors[level] = weights

    write_csv(output / "metrics.csv", summaries)

    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output / "feature_weights.csv",
        [
            {
                "feature": name,
                **{
                    level: weight_vectors[level][i]
                    for level in LEVELS
                },
            }
            for i, name in enumerate(Solver.FEATURE_NAMES)
        ],
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 13))

    for ax, level in zip(axes, LEVELS):
        for algorithm in ("ql", "fq"):
            history = runs[(level, algorithm)]

            ax.plot(
                [row["episode"] for row in history],
                [row["moving_average_50"] for row in history],
                label=algorithm.upper(),
                linewidth=1.2,
            )

        ax.set(
            title=f"{level}: official per-algorithm budgets",
            xlabel="Training episode",
            ylabel="Mean reward (50 episodes)",
        )

        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        output / "baseline_learning_curves.png",
        dpi=180,
    )
    plt.close(fig)

    print(f"Saved results to: {output}", flush=True)


if __name__ == "__main__":
    main()