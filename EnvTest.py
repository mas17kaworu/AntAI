import numpy as np


class AntEnv:
    def __init__(self):
        self.observation_space_shape = 400
        self.action_space_num = 5
        cols = 20
        rows = 20
        self.stepNum = 0
        self.DONE = True
        self.state = [0 for i in range(cols*rows)]
        self.s1 = np.array(self.state)

    def reset(self):
        self.DONE = False
        return self.s1

    def step(self, action):
        reward = 0
        if action == 0:
            reward = 10
        self.stepNum += 1
        if self.stepNum > 100:
            self.DONE = True
        return self.s1, reward, self.DONE, 'xxx'
