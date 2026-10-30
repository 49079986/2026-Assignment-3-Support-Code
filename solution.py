import sys
import time

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 16/10/2026
"""

"""
COMP3702 Assignment 3: CrystalRover

AI assistance: OpenAI Codex assisted with the implementation.
"""

from collections import deque
import math
import random

from game_env import GameEnv
from game_state import GameState


class Solver:
    STUDENT_NAME = "Yuyang Gong"
    STUDENT_ID = "49079984"
    GITHUB_USERNAME = "49079986"

    # Tabular Q-learning learning rate.
    QL_ALPHA = 0.2

    # Feature-based Q-learning learning rate.
    FQ_ALPHA = 0.03

    # Exploration schedule.
    EPSILON_START = 0.25
    EPSILON_END = 0.02
    EPSILON_DECAY_FRACTION = 0.8

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env

    @staticmethod
    def testcases_to_attempt():
        return [1, 2, 3, 4, 5]

    def ql_initialise(self):
        # Map each GameState to one Q-value per action.
        self.q_table = {}

        # Store episode statistics for later experiments.
        self.ql_history = []

        self.ql_alpha = self.QL_ALPHA

    def ql_train_episode(self):
        env = self.game_env
        state = env.get_init_state()

        total_reward = 0.0
        steps = 0
        epsilon = self._epsilon()

        for _ in range(env.max_episode_steps):
            if env.is_terminal(state):
                break

            row = self.q_table.setdefault(
                state, [0.0] * len(env.ACTIONS)
            )

            valid_actions = env.get_valid_actions(state)
            action = self._training_action(
                valid_actions, row, epsilon
            )
            action_index = env.ACTIONS.index(action)

            next_state, reward, _ = env.perform_action(
                state, action
            )

            target = reward

            # Terminal states have no future reward term.
            if not env.is_terminal(next_state):
                next_row = self.q_table.get(next_state)

                if next_row is not None:
                    next_best = max(
                        next_row[env.ACTIONS.index(a)]
                        for a in env.get_valid_actions(next_state)
                    )
                    target += env.gamma * next_best

            row[action_index] += self.ql_alpha * (
                target - row[action_index]
            )

            # Continue from the state returned by the environment.
            state = next_state
            total_reward += reward
            steps += 1

            if (
                env.is_terminal(state)
                or total_reward <= env.ql_reward_min_tgt
                or env.is_training_budget_exhausted()
            ):
                break

        self._record_episode(
            self.ql_history,
            total_reward,
            steps,
            epsilon,
            state,
        )

    def _epsilon(self):
        env = self.game_env
        budget = env.training_step_budget or 1

        decay_steps = max(
            1.0, budget * self.EPSILON_DECAY_FRACTION
        )
        progress = min(
            1.0, env.training_steps_used / decay_steps
        )

        return self.EPSILON_START + progress * (
            self.EPSILON_END - self.EPSILON_START
        )

    def _training_action(self, valid_actions, row, epsilon):
        if random.random() < epsilon:
            return random.choice(valid_actions)

        best_value = max(
            row[self.game_env.ACTIONS.index(a)]
            for a in valid_actions
        )

        best_actions = [
            a for a in valid_actions
            if row[self.game_env.ACTIONS.index(a)] == best_value
        ]

        return random.choice(best_actions)

    def _record_episode(self, history, reward, steps, epsilon, state):
        history.append({
            "episode": len(history) + 1,
            "training_steps": self.game_env.training_steps_used,
            "reward": reward,
            "steps": steps,
            "epsilon": epsilon,
            "solved": self.game_env.is_solved(state),
            "terminal": self.game_env.is_terminal(state),
        })

    def ql_select_action(self, state: GameState):
        valid_actions = self.game_env.get_valid_actions(state)
        q_values = self.q_table.get(state)

        # Unseen states have implicit Q-values of zero.
        # Choose deterministically without modifying the table.
        if q_values is None:
            return valid_actions[0]

        return max(
            valid_actions,
            key=lambda action: q_values[
                self.game_env.ACTIONS.index(action)
            ],
        )

    def fq_initialise(self):
        pass

    def fq_get_features(self, state: GameState, action):
        pass

    def fq_train_episode(self):
        pass

    def fq_select_action(self, state: GameState):
        pass