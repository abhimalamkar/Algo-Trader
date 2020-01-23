import gym
from gym import error, spaces, utils
from gym.utils import seeding
import enum
import numpy as np
import random

random.seed(0)

DEFAULT_COMMISSION_PERC = 0.00
DEFAULT_WINDOW_SIZE = 300
PAGE = 0
TRADIN_INTERVAL = 100
SHOULD_USE_INTERVAL = True

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, test=False):

        self._test_env = test
        self._prices = prices
        self._initial_offset = PAGE * TRADIN_INTERVAL#random.randint(DEFAULT_WINDOW_SIZE,len(self._prices) - 1)
        self._offset = self._initial_offset
        self._window_size = DEFAULT_WINDOW_SIZE
        self._commission_perc = DEFAULT_COMMISSION_PERC 

        self.have_position = False
        self.open_price = 0

        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(DEFAULT_WINDOW_SIZE,1), dtype=np.float32)

    def reset(self):
        self._initial_offset = PAGE * TRADIN_INTERVAL #random.randint(DEFAULT_WINDOW_SIZE,len(self._prices) - 1)
        self._offset = self._initial_offset        
        self.have_position = False
        return self.get_state(self._offset,self._window_size + 1)

    def get_state(self, t, n):
        d = t - n + 1
        block = self._prices[d : t + 1] if d >= 0 else -d * [self._prices[0]] + self._prices[0 : t + 1]
        res = []

        for i in range(n - 1):
            try:
                res.append(((block[i + 1] - block[i]) / block[i]) * 100 )
            except:
                res.append(0)
        return np.array(res)

    def _cur_close(self):
        return self._prices[self._offset]

    def _set_seed(self,seed):
        random.seed(seed)

    def take_action(self,action):
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self._commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self._commission_perc
            reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        done = self._offset >= len(self._prices) or (SHOULD_USE_INTERVAL and self._offset >= self._initial_offset + TRADIN_INTERVAL)

        if self._test_env and done:
            print("Total percentage change " ,((self._prices[-1] - self._prices[self._initial_offset]) / self._prices[self._initial_offset]) * 100)

        # if done:
        #     print("Reward before :" + str(reward))
        #     reward -= ((self._prices[self._offset] - self._prices[self._initial_offset]) / self._prices[self._initial_offset]) * 100
        #     print("Reward after :" + str(reward))

        return reward, done

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self.take_action(action)
        obs = self.get_state(self._offset,self._window_size + 1)
        info = { "offset": self._offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

