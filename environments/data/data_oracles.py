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

"""Oracles (experts) for data tasks."""

import random

from ibc.environments.data import data
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class DataOracle(py_policy.PyPolicy):
  """Oracle moving between two different goals."""

  def __init__(self,
               env):
    """Create oracle.

    Args:
      env: Environment.
    """
    super(DataOracle, self).__init__(env.time_step_spec(),
                                         env.action_spec())
    self._env = env
    self.reset()

  def reset(self):
    pass

  def _action(self, time_step,  # pytype: disable=signature-mismatch  # re-none
              policy_state):
    obs = time_step.observation
    act = np.copy(obs['pos_first_goal'])

    return policy_step.PolicyStep(action=act)
