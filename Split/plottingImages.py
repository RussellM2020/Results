from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img_size = 20
mid = int(img_size/2)
full_grid = np.ones((img_size,img_size,3))
full_grid[mid,mid] = np.array([0, 0, 1])
#full_grid[int(self._state[0]) + self.mid, int(self._state[1]) + self.mid] = np.array([0, 1, 0])
print(full_grid)
#plt.imshow(full_grid)
plt.show()