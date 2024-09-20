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

"""Evaluates TF-Agents policies."""
import functools
import os
import shutil

from absl import app
from absl import flags
from absl import logging

import gin
# # Need import to get env resgistration.
from ibc.environments.data import data  # pylint: disable=unused-import
from ibc.environments.data import data_oracles
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.metrics import py_metrics
# Need import to get tensorflow_probability registration.
from tf_agents.utils import example_encoding_dataset
from gym.envs.registration import registry
import pandas as pd


def evaluate(num_episodes, dataset_path=None,):
  """Evaluates the given policy for n episodes."""
  env_name = 'Data-v0'
  env = suite_gym.load(env_name)
  policy = data_oracles.DataOracle(env)

  metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
  ]

  observers = metrics[:]

  if dataset_path:
    # TODO(oars, peteflorence): Consider a custom observer to filter only
    # positive examples.
    observers.append(
        example_encoding_dataset.TFRecordObserver(
            dataset_path,
            policy.collect_data_spec,
            py_mode=True,
            compress_image=False))

  driver = py_driver.PyDriver(env, policy, observers, max_episodes=num_episodes)
  time_step = env.reset()
  initial_policy_state = policy.get_initial_state(1)
  driver.run(time_step, initial_policy_state)
  log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
  logging.info('\n\t\t '.join(log))

  env.close()

if __name__ == "__main__":
    evaluate(
        num_episodes=1000,
        dataset_path='/app/test_set_pt/20220630_L4c-D8_l2cs/output.tfrecord',
    )