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


@gin.configurable
class GazeEnv(gym.Env):
  """Simple gaze environment with gym wrapper.

  A key feature of this environment is that it is N-dimensional, i.e. the
  observation space is:
    3N:
      - position of the facilitator (N dimensions)
      - position of the participants (N dimensions)
      - position of the goal (N dimensions)
  
  and action space is:
    N:
      - next position of the facilitator (N dimensions)

  Key functions:
  - reset() --> state
  - step(action) --> state, reward, done, info
  """

  @gin.configurable
  def __init__(
      self,
      n_steps = 50,
      n_dim = 2,
      seed = None,
      dt = 0.005,  # 0.005 = 200 Hz
      repeat_actions = 10,  # 10 makes control 200/10 = 20 Hz
      k_p = 10.,
      k_v = 5.,
      goal_distance = 0.01,
      eval = False,
      eval_dir = None,
  ):
    """Creates an env instance with options, see options below.

    Args:
      n_steps: # of steps until done.
      n_dim: # of dimensions.
      seed: random seed.
      dt: timestep for internal simulation (not same as control rate)
      repeat_actions: repeat the action this many times, each for dt.
      k_p: P gain in PD controller. (p for position)
      k_v: D gain in PD controller. (v for velocity)
      goal_distance: Acceptable distances to goals for success.
    """
    self.reset_counter = 0
    self.img_save_dir = None

    self.n_steps = n_steps
    self.goal_distance = goal_distance

    # If we wanna eval the model correctly
    self.eval = eval
    self.eval_dir = eval_dir
    if self.eval:
      self.df = pd.read_csv(self.eval_dir)

    self.count = 0
    self.steps = 0
    self.n_dim = n_dim
    # self.hide_velocity = hide_velocity
    self._rng = np.random.RandomState(seed=seed)

    self.dt = dt
    self.repeat_actions = repeat_actions
    # Make sure is a multiple.
    assert int(1/self.dt) % self.repeat_actions == 0

    self.k_p = k_p
    self.k_v = k_v
    self.action_space = spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(self.n_dim,),
                                   dtype=np.float32)
    self.observation_space = self._create_observation_space()
    self.reset()
  
  def get_metrics(self, num_episodes):
    metrics = [
        gaze_metrics.AverageFirstGoalDistance(
            self, buffer_size=num_episodes),
        gaze_metrics.AverageSuccessMetric(
            self, buffer_size=num_episodes)
    ]
    success_metric = metrics[-1]
    return metrics, success_metric

  def _create_observation_space(self):
    obs_dict = collections.OrderedDict(
      fac_gaze_agent=spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(self.n_dim,), dtype=np.float32),
      vel_gaze_agent=spaces.Box(low=-1e2, high=1e2, shape=(self.n_dim,), dtype=np.float32),
      pos_first_goal=spaces.Box(low=-np.pi/2., high=np.pi/2, shape=(self.n_dim,), dtype=np.float32),
      all_gaze_agent=spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(12,), dtype=np.float32),
    )
    

    # if self.hide_velocity:  # pytype: disable=attribute-error
    #   del obs_dict['vel_agent']

    return spaces.Dict(obs_dict)
  
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

  def seed(self, seed=None):
    self._rng = np.random.RandomState(seed=seed)

  def reset(self):
    self.reset_counter += 1  # pytype: disable=attribute-error
    if self.eval:
      if self.reset_counter > 2:
        self.steps += 1
        self.count += 1
    else:
      self.steps = 0
    # self.obs_log and self.act_log hold internal state,
    # will be useful for plotting.
    self.obs_log = []
    self.act_log = []
    self.new_actions = []

    obs = dict()
    if self.eval:
      row = self.df.iloc[self.steps]
      self._update_state_from_csv_row(row)
      df_obs = self.obs_log[-1]
      obs['fac_gaze_agent'] = df_obs['fac_gaze_agent']  # pytype: disable=attribute-error
      obs['vel_gaze_agent'] = np.zeros((self.n_dim)).astype(np.float32)  # pytype: disable=attribute-error
      obs['pos_first_goal'] = df_obs['pos_first_goal']  # pytype: disable=attribute-error
      obs['all_gaze_agent'] = df_obs['all_gaze_agent']  # pytype: disable=attribute-error
    else:
      obs['fac_gaze_agent'] = self._rng.rand(self.n_dim).astype(np.float32)  # pytype: disable=attribute-error
      obs['vel_gaze_agent'] = np.zeros((self.n_dim)).astype(np.float32)  # pytype: disable=attribute-error
      obs['pos_first_goal'] = self._rng.rand(self.n_dim).astype(np.float32)  # pytype: disable=attribute-error
      obs['all_gaze_agent'] = self._rng.rand(12).astype(np.float32)  # pytype: disable=attribute-error
    
    self.obs_log.append(obs)

    self.min_dist_to_first_goal = np.inf

    return self._get_state()

  def _get_state(self):
    return copy.deepcopy(self.obs_log[-1])  # pytype: disable=attribute-error

  def _internal_step(self, action, new_action):
    if new_action:
      self.new_actions.append(len(self.act_log))  # pytype: disable=attribute-error
    self.act_log.append({'pos_setpoint': action})  # pytype: disable=attribute-error
    obs = self.obs_log[-1]  # pytype: disable=attribute-error
    # u = k_p (x_{desired} - x) + k_v (xdot_{desired} - xdot)
    # xdot_{desired} is always (0, 0) -->
    # u = k_p (x_{desired} - x) - k_v (xdot)
    u_agent = self.k_p * (action - obs['fac_gaze_agent']) - self.k_v * (  # pytype: disable=attribute-error
        obs['vel_gaze_agent'])
    new_xy_agent = obs['fac_gaze_agent'] + obs['vel_gaze_agent'] * self.dt  # pytype: disable=attribute-error
    new_velocity_agent = obs['vel_gaze_agent'] + u_agent * self.dt  # pytype: disable=attribute-error
    obs = copy.deepcopy(obs)
    obs['fac_gaze_agent'] = new_xy_agent
    obs['vel_gaze_agent'] = new_velocity_agent
    self.obs_log.append(obs)  # pytype: disable=attribute-error

  def dist(self, goal):
    current_position = self.obs_log[-1]['fac_gaze_agent']  # pytype: disable=attribute-error
    return np.linalg.norm(current_position - goal)

  def _get_reward(self, done):
    """Reward is 1.0 if agent hits goal."""

    # This also statefully updates these values.
    self.min_dist_to_first_goal = min(
        self.dist(self.obs_log[0]['pos_first_goal']),  # pytype: disable=attribute-error
        self.min_dist_to_first_goal)  # pytype: disable=attribute-error

    def _reward(thresh):
      reward_first = True if self.min_dist_to_first_goal < thresh else False
      return 1.0 if (reward_first and done) else 0.0

    reward = _reward(self.goal_distance)  # pytype: disable=attribute-error
    return reward

  @property
  def succeeded(self):
    thresh = self.goal_distance  # pytype: disable=attribute-error
    hit_first = True if self.min_dist_to_first_goal < thresh else False

    return hit_first

  def step(self, action):
    self.steps += 1
    self.count += 1

    self._internal_step(action, new_action=True)
    for _ in range(self.repeat_actions - 1):  # pytype: disable=attribute-error
      self._internal_step(action, new_action=False)
    state = self._get_state()

    if self.count >= self.n_steps:
      self.count = 0
      done = True  
    else:
      done = False  # pytype: disable=attribute-error

    reward = self._get_reward(done)
    return state, reward, done, {}


# Make sure we only register once to allow us to reload the module in colab for
# debugging.
if 'Gaze-v0' in registration.registry.env_specs:
  del registration.registry.env_specs['Gaze-v0']

registration.register(id='Gaze-v0', entry_point=GazeEnv)
