"""
Credits: https://github.com/zuoxingdong/dm2gym/blob/master/dm2gym/envs/dm_suite_env.py
"""

import gym
from gym import spaces
import park 
import park.core
import park.spaces
import gym.spaces
import numpy as np


def convert_park_to_gym_space(park_space: park.core.Space):
    r"""Convert park space to gym space. """
    if isinstance(park_space, park.spaces.Box):
        return gym.spaces.Box(low=park_space.low, high=park_space.high, dtype=park_space.dtype, shape=park_space.shape)
    elif isinstance(park_space, park.spaces.Discrete):
        return gym.spaces.Discrete(n=park_space.n)
    elif isinstance(park_space, park.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=park_space.nvec, dtype=np.int64)


class ParkWrapper(gym.Env):
    def __init__(self, env, env_name):
        self.env = env
        self.observation_space = convert_park_to_gym_space(self.env.observation_space)
        self.action_space = convert_park_to_gym_space(self.env.action_space)
        self._is_park_env = True
        self.env_name = env_name

        if hasattr(self.env, 'metadata'):
            self.metadata = self.env.metadata
        if hasattr(self.env, 'reward_range'):
            self.reward_range = self.env.reward_range
    
    def seed(self, seed):
        return self.env.seed(seed)
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
