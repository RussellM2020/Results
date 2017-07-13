from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class PointImageEnv(Env):
    #TODO: Need to make this images

    def __init__(self, img_size=40):
        self.img_size = img_size
        self.mid = int(self.img_size/2)
        super(Env, self).__init__()

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.img_size, self.img_size, 3)) #TODO: Fix the shape to be correct

    @property
    def action_space(self):
        return Box(low=-2, high=2, shape=(2,))

    def reset(self):
        self._state = np.random.uniform(-self.mid + 1, self.mid - 1, size=(2,))
        observation = self.get_image() #TODO: Get image here
        return observation

    def get_image(self): 
        full_grid = np.ones((self.img_size,self.img_size,3))
        full_grid[self.mid,self.mid] = np.array([0, 0, 1])
        full_grid[int(self._state[0]) + self.mid, int(self._state[1]) + self.mid] = np.array([0, 1, 0])
        return full_grid #Simulating image with a red goal and a green object

    def step(self, action):
        self._state = self._state + action #Need to clips
        self._state = np.clip(self._state, -self.mid + 1, self.mid - 1)
        x, y = self._state
        dist = x ** 2 + y ** 2
       
        #Square of side 40. by square(s) - pi*square(r) = pi*square(r), we get expected value of r approx. 16
        # This is confirmed by emperical observation
        #avgDist = 16**2
        #leverageFactor = 10**2
        #reward = -dist  
        reward = -dist
        #+ math.exp(-dist)*avgDist*leverageFactor
        # for dist 0, 25600
        #print()
        done = False
        if dist<=0.1:
            done = True
        #abs(x) < 0.01 and abs(y) < 0.01
        next_observation = self.get_image()
        return Step(observation=next_observation, reward=reward, done=done, state = self._state)

