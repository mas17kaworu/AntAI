
class AntEnv:
    def __init__(self):
        self.observation_space_shape = 400
        self.action_space_num = 5
        cols = 20
        rows = 20
        self.stepNum = 0
        self.DONE = True
        self.state = [0 for col in range(cols*rows)]

    def reset(self):
        self.DONE = False
        return self.state

    def step(self, action):
        self.stepNum += 1
        if stepNum > 100:
            self.DONE = true
        return self.state, 10, self.DONE, 'xxx'
