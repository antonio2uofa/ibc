# coding=utf-8
# Copyright 2024 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracles (experts) for particle tasks."""

import random

from ibc.environments.gaze import gaze
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class GazeOracle(py_policy.PyPolicy):
    """Oracle moving to a single goal (pos_first_goal)."""

    def __init__(self, env, wait_at_first_goal=1, goal_threshold=0.01):
        """Create oracle.

        Args:
          env: Environment.
          wait_at_first_goal: How long to wait at first goal, once you get there.
                              Encourages memory.
          goal_threshold: How close is considered good enough.
        """
        super(GazeOracle, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)

        assert wait_at_first_goal > 0
        self.wait_at_first_goal = wait_at_first_goal
        assert goal_threshold > 0.
        self.goal_threshold = goal_threshold

        self.reset()

    def reset(self):
        """Reset oracle."""
        self.steps_at_first_goal = 0

    def _action(self, time_step, policy_state):
        """Determine action based on observation and goal proximity."""
        if time_step.is_first():
            self.reset()

        obs = time_step.observation
        gt_goal = self._env.obs_log[0]['pos_first_goal']  # Only one goal
        dist = np.linalg.norm(obs['fac_gaze_agent'] - gt_goal)

        if dist < self.goal_threshold:
            self.steps_at_first_goal += 1

        # Continue moving to the first goal
        act = np.copy(gt_goal)

        return policy_step.PolicyStep(action=act)
