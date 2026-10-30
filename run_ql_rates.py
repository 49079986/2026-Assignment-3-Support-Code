"""
A3 fixed-alpha experiments using the official harness.

AI assistance: OpenAI Codex, 30 October 2026.
"""

from bisect import bisect_left
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_env import GameEnv, TrainingBudgetExhausted
from harness import (
    evaluate_final_policy,
    get_learning_threshold,
    seed_solver_randomness,
    train_agent,
)
from solution import Solver


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.1, 0.2, 0.4)


def moving_average(values, window=50):
    means = []
    total = 0.0

    for i, value in enumerate(values):
        total += value

        if i >= window:
            total -= values[i - window]

        means.append(total / min(i + 1, window))

    return means


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def run_one(alpha, output):
    testcase = str(ROOT / "testcases" / "L1.txt")

    training_env = GameEnv(testcase)
    control_env = GameEnv(testcase)

    training_env.set_training_budget(
        training_env.ql_training_step_budget,
        training_env.ql_training_reward_budget,
    )

    # Reset randomness before every independent run.
    seed_solver_randomness(control_env.episode_seed)

    solver = Solver(training_env)
    solver.QL_ALPHA = alpha

    result = train_agent(
        training_env,
        control_env,
        solver,
        "ql",
        TrainingBudgetExhausted,
    )

    mean_reward, solved = evaluate_final_policy(
        control_env,
        solver.ql_select_action,
        "ql",
    )

    means = moving_average(
        [row["reward"] for row in solver.ql_history]
    )

    history = [
        dict(row, moving_average_50=mean)
        for row, mean in zip(solver.ql_history, means)
    ]

    if not history:
        raise RuntimeError("No training episodes were recorded")

    write_csv(
        output / f"alpha_{alpha:g}_episodes.csv",
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
            output / f"alpha_{alpha:g}_checkpoints.csv",
            checkpoints,
        )

    learned_episode = None

    if result.learned_at_steps is not None:
        learned_episode = bisect_left(
            episode_steps, result.learned_at_steps
        ) + 1

    summary = {
        "level": "L1",
        "alpha": solver.ql_alpha,
        "seed": control_env.episode_seed,
        "gamma": control_env.gamma,
        "max_episode_steps": control_env.max_episode_steps,
        "epsilon_start": solver.EPSILON_START,
        "epsilon_end": solver.EPSILON_END,
        "epsilon_decay_fraction": solver.EPSILON_DECAY_FRACTION,
        "step_budget": training_env.training_step_budget,
        "reward_budget": training_env.ql_training_reward_budget,
        "training_steps": result.steps_used,
        "training_reward": result.training_reward,
        "training_episodes": len(history),
        "harness_episode_calls": result.episodes,
        "learning_threshold": get_learning_threshold(
            control_env, "ql"
        ),
        "learned_at_steps": result.learned_at_steps,
        "learned_at_episodes": learned_episode,
        "evaluation_reward_floor": control_env.ql_reward_min_tgt,
        "evaluation_mean_reward": mean_reward,
        "solved_trials": solved,
        "evaluation_trials": 1000,
        "time_per_training_step": result.get_time_per_step(),
        "solver_sha256": hashlib.sha256(
            (ROOT / "solution.py").read_bytes()
        ).hexdigest(),
    }

    print(
        f"alpha={alpha:g}: {len(history)} episodes, "
        f"{result.steps_used} steps, "
        f"mean reward={mean_reward:.2f}, "
        f"solved={solved}/1000",
        flush=True,
    )

    return history, summary


def main():
    output = ROOT / "results" / (
        "ql_rates_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    runs = []
    summaries = []

    for alpha in ALPHAS:
        history, summary = run_one(alpha, output)
        runs.append((alpha, history))
        summaries.append(summary)

    write_csv(output / "metrics.csv", summaries)

    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    # Compare the three moving-average learning curves.
    fig, ax = plt.subplots(figsize=(10, 5))

    for alpha, history in runs:
        ax.plot(
            [row["episode"] for row in history],
            [row["moving_average_50"] for row in history],
            label=f"alpha={alpha:g}",
            linewidth=1.2,
        )

    ax.set(
        xlabel="Training episode",
        ylabel="Mean reward (50 episodes)",
        title="L1: fixed learning-rate comparison",
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output / "learning_rates.png", dpi=180)
    plt.close(fig)

    # Save raw rewards alongside their moving averages.
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    for ax, (alpha, history) in zip(axes, runs):
        episodes = [
            row["episode"] for row in history
        ]

        ax.plot(
            episodes,
            [row["reward"] for row in history],
            alpha=0.25,
            linewidth=0.5,
            label="Episode reward",
        )

        ax.plot(
            episodes,
            [row["moving_average_50"] for row in history],
            linewidth=1.2,
            label="50-episode moving average",
        )

        ax.set(
            title=f"L1: alpha={alpha:g}",
            xlabel="Training episode",
            ylabel="Reward",
        )
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output / "raw_rewards.png", dpi=180)
    plt.close(fig)

    print(f"Saved results to: {output}", flush=True)


if __name__ == "__main__":
    main()