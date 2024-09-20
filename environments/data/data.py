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

"""Simple particle environment."""

import collections
import copy
import os
from typing import Union

import gin
import gym
from gym import spaces
from gym.envs import registration
from ibc.environments.gaze import gaze_metrics
from ibc.environments.gaze import gaze_viz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import csv

class DataEnv(gym.Env):
    """Simple data environment reading from a CSV file."""

    def __init__(
        self,
        csv_file,
        n_steps=50,
        n_dim=2,
        dt=1/30,
        seed=None,
    ):
        super(DataEnv, self).__init__()
        
        self.csv_file = csv_file
        self.n_steps = n_steps
        self.n_dim = n_dim
        self.dt = dt
        self.steps = 0
        self.count = 0
        
        self.action_space = spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(self.n_dim,), dtype=np.float32)
        self.observation_space = self._create_observation_space()
        
        self._rng = np.random.RandomState(seed=seed)
        self._df = pd.read_csv(csv_file)
        self.current_row = self._df.iloc[self.steps]
        
        self.reset()
        
    def _create_observation_space(self):
        return spaces.Dict({
            'fac_gaze_agent': spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(self.n_dim,), dtype=np.float32),

            'vel_gaze_agent': spaces.Box(low=-1e2, high=1e2, shape=(self.n_dim,), dtype=np.float32),

            'pos_first_goal': spaces.Box(low=-np.pi/2., high=np.pi/2, shape=(self.n_dim,), dtype=np.float32),

            'all_gaze_agent': spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(12,), dtype=np.float32),
        })

    def reset(self):
        self.obs_log = []
        self.act_log = []
        self.new_actions = []
        self.count += 1
        self.steps += 1
        # Initialize state with the first row of CSV
        if self.steps < len(self._df):
            self.current_row = self._df.iloc[self.steps]
        self._update_state_from_csv_row(self.current_row)
        return self._get_state()

    def _update_state_from_csv_row(self, row):
        # Extract each observation
        fac_gaze_agent = np.array([float(row['obs_fac_x']), float(row['obs_fac_y'])], dtype=np.float32)
        vel_agent = np.array([float(row['velocity_x']), float(row['velocity_y'])], dtype=np.float32)
        pos_first_goal = np.array([float(row['target_x']), float(row['target_y'])], dtype=np.float32)
        
        # Extract and concatenate all gaze observations
        all_gaze_agent = np.array([
            float(row['obs_1_x']), float(row['obs_1_y']),
            float(row['obs_2_x']), float(row['obs_2_y']),
            float(row['obs_3_x']), float(row['obs_3_y']),
            float(row['obs_4_x']), float(row['obs_4_y']),
            float(row['obs_5_x']), float(row['obs_5_y']),
            float(row['obs_fac_x']), float(row['obs_fac_y'])
        ], dtype=np.float32)
        
        # Update the observation log
        obs = {
            'fac_gaze_agent': fac_gaze_agent,
            'vel_gaze_agent': vel_agent,
            'pos_first_goal': pos_first_goal,
            'all_gaze_agent': all_gaze_agent,
        }

        self.obs_log.append(obs)

    
    def _get_state(self):
        return copy.deepcopy(self.obs_log[-1])

    def step(self, action):
        self.count += 1 
        if self.count == self.n_steps:
            self.count = 0
            done = True
        else: 
            self.steps += 1
            done = False
            # Store action
            self.new_actions.append(len(self.act_log))
            self.act_log.append({'pos_setpoint': action})

            if self.steps < len(self._df):
                self.current_row = self._df.iloc[self.steps]
                self._update_state_from_csv_row(self.current_row)
            else:
                done = True
        
        state = self._get_state()
        reward = self._get_reward(done)
        return state, reward, done, {}

    def _get_reward(self, done):
        """Custom reward function."""
        # Implement reward calculation based on the specific needs of your environment
        if done:
            return 1.0
        return 0.0


# Make sure we only register once to allow us to reload the module in colab for
# debugging.
if 'Data-v0' in registration.registry.env_specs:
  del registration.registry.env_specs['Data-v0']

registration.register(id='Data-v0', entry_point=DataEnv, kwargs={'csv_file': '/app/test_set_pt/20220630_L4c-D8_l2cs/output.csv'})
