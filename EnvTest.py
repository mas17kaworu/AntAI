import numpy as np
import os
Start_play_command = 'D:\Python27\python tools/playgame.py "python MyBot_1.py" "python tools/sample_bots/python/HunterBot.py"  ' \
                     '--map_file "tools/maps/example/tutorial1.map" --log_dir %s --turns 60 --scenario  --player_seed 7 --nolaunch -e'


class AntEnv:
    def __init__(self, name):
        self.Env_name = name
        self.observation_space_shape = 400
        self.action_space_num = 5
        cols = 20
        rows = 20
        self.stepNum = 0
        self.DONE = True
        self.state = [0 for _ in range(cols*rows)]
        self.s1 = np.array(self.state)

    def reset(self):
        self.DONE = False
        command = Start_play_command % ('ant_log_' + self.Env_name)
        print(command)
        os.system(command)
        return self.s1

    def step(self, action):
        reward = 0
        if action == 0:
            reward = 10
        self.stepNum += 1
        if self.stepNum > 100:
            self.DONE = True
        return self.s1, reward, self.DONE, 'xxx'
