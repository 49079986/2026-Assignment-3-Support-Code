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
        pass

    def ql_train_episode(self):
        pass

    def ql_select_action(self, state: GameState):
        pass

    def fq_initialise(self):
        pass

    def fq_get_features(self, state: GameState, action):
        pass

    def fq_train_episode(self):
        pass

    def fq_select_action(self, state: GameState):
        pass