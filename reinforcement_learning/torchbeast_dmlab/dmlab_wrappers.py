# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from collections import deque
import deepmind_lab


# 9 actions, each encoded by a 7-dim vector.
DEFAULT_ACTION_SET_TUPLE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

DEFAULT_ACTION_SET = np.array(DEFAULT_ACTION_SET_TUPLE, dtype=np.intc)


# Based on https://github.com/deepmind/scalable_agent/blob/master/environments.py
class create_env_dmlab(object):
  """Wrapper around DMLab-30 env."""

  def __init__(self, level, config, seed, skip=4,
               runfiles_path=None, level_cache=None):
    self._skip = skip
    self.obs_shape = (3, 72, 96)  # reshape from (72, 96, 3) RGB_INTERLEAVED
    self._random_state = np.random.RandomState(seed=seed)
    if runfiles_path:
      deepmind_lab.set_runfiles_path(runfiles_path)
    config = {k: str(v) for k, v in config.items()}
    self._observation_spec = ['RGB_INTERLEAVED']
    self._env = deepmind_lab.Lab(
        level=level,
        observations=self._observation_spec,
        config=config,
        level_cache=level_cache,
    )
    self._obs_buffer = np.zeros((2,)+self.obs_shape, dtype=np.uint8)

  # Minimum required (see structure/expected return object):
  # def reset(self):
  #     print("reset called")
  #     return np.ones((4, 84, 84), dtype=np.uint8)

  # def step(self, action):
  #     frame = np.zeros((4, 84, 84), dtype=np.uint8)
  #     return frame, 0.0, False, {}  # First three mandatory.

  def reset(self):
    # return observation!
    self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
    d = self._env.observations()
    return d['RGB_INTERLEAVED'].transpose(2, 0, 1)

  def get_obs(self):
    d = self._env.observations()
    return d['RGB_INTERLEAVED'].transpose(2, 0, 1)

  # atari style max pooling-based frame skipping
  def step(self, action):
    # `action` is an index here
    action_code = DEFAULT_ACTION_SET[action]
    total_reward = 0.0
    done = None

    for i in range(self._skip):
        reward = self._env.step(action_code)
        total_reward += reward
        done = not self._env.is_running()
        if done:
            max_frame = self.reset()
            return max_frame, total_reward, done, {}

        obs = np.array(self.get_obs(),dtype=np.uint8)
        if i == self._skip - 2: self._obs_buffer[0] = obs
        if i == self._skip - 1: self._obs_buffer[1] = obs

    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, {}

#  def step(self, action):
#    # `action` is an index here
#    action_code = DEFAULT_ACTION_SET[action]
#    reward = self._env.step(action_code)
#
#    done = not self._env.is_running()
#    if done:
#        self.reset()
#
#    observation = np.array(self.get_obs(),dtype=np.uint8)
#    return observation, reward, done, {}

  def close(self):
    self._env.close()
