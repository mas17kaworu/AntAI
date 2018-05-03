import numpy as np
import pickle
import os
import socket
from queue import Queue
import threading
import antLog
import Constants

Start_play_command = 'D:\Python27\python tools/playgame.py "python %s" "python tools/sample_bots/python/HunterBot.py"  ' \
                     '--map_file "tools/maps/example/tutorial1.map" --log_dir %s --turns 100 --scenario  --nolaunch' \
                     ' --player_seed 7  --turntime 5000 -e'
# --verbose   --nolaunch
# map:  maze/maze_02p_02.map

PORT1 = 12023
PORT2 = 12025
PORT3 = 12024
PORT4 = 12041
PORT5 = 12026
PORT6 = 12027




class AntEnv:
    def __init__(self, name):
        self.Env_name = name
        self.observation_space_shape = Constants.MAP_HEIGHT * Constants.MAP_WIDTH   # 43*39=1677   +2
        self.action_space_num = 5  # Action space  # 0--stay 1--North 2--East 3--South 4--West
        cols = 20
        rows = 20
        self.stepNum = 1
        self.DONE = False
        self.state = [0 for _ in range(cols*rows)]
        self.ants_loc_queue = Queue()
        self.ants_loc_list = []
        self.state_queue = Queue()
        self.connection = None

    def reset(self):
        command = ''
        self.stepNum = 1
        self.DONE = False
        self.state_queue.empty()
        self.ants_loc_queue.empty()
        if self.Env_name == 'W_1':
            command = Start_play_command % ('MyBot_1.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT1,))
            t.start()
            print("Bot1 reStart")
        elif self.Env_name == 'W_2':
            command = Start_play_command % ('MyBot_2.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT2,))
            t.start()
            print("Bot2 reStart")
        elif self.Env_name == 'W_3':
            command = Start_play_command % ('MyBot_3.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT3,))
            t.start()
            print("Bot3 reStart")
        elif self.Env_name == 'W_4':
            command = Start_play_command % ('MyBot_4.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT4,))
            t.start()
            print("Bot4 reStart")
        elif self.Env_name == 'W_5':
            command = Start_play_command % ('MyBot_5.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT5,))
            t.start()
            print("Bot4 reStart")
        elif self.Env_name == 'W_6':
            command = Start_play_command % ('MyBot_6.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT6,))
            t.start()
            print("Bot4 reStart")
        os.popen(command)
        tmp_state = []
        self.ants_loc_list = []
        try:
            tmp_state = self.state_queue.get(block=True, timeout=5)
            tmp_state = np.array(tmp_state)
            antLog.write_log('receive state ', self.Env_name)
            self.ants_loc_list = self.ants_loc_queue.get(block=True, timeout=6)
            antLog.write_log('receive ants ', self.Env_name)
        except Exception as err:
            self.connection.close
            antLog.write_log('receive exception in reset', self.Env_name)
            self.DONE = True
        self.state = tmp_state
        return tmp_state, self.ants_loc_list, self.DONE

    def step(self, actions):

        next_state = None
        next_ants = None
        antLog.write_log('send to Ant %s' % str(actions), self.Env_name)
        self.connection.sendall(pickle.dumps(actions, protocol=2))
        try:
            next_state = self.state_queue.get(timeout=5)
            next_state = np.array(next_state)
            antLog.write_log('receive state ', self.Env_name)
            next_ants = self.ants_loc_queue.get(timeout=1)
            antLog.write_log('receive Ants ', self.Env_name)
        except Exception as err:
            self.DONE = True

        # print("next_ants = ", next_ants)
        # print("shape" + str(next_state.shape))
        if not self.DONE:
            # increase = len(next_ants) - (len(actions)/2)
            # if increase == 0:
            #     reward = 1
            # elif increase > 0:
            #     reward = increase * 3
            # else:
            #     reward = increase * 2

            ant_rewards, loc_dict = self.generate_ant_reward(actions=actions, map_state_next=next_state)

            self.stepNum += 1
        else:
            loc_dict = {}
            ant_rewards = {(0, 0): 0}

        self.ants_loc_list = next_ants
        self.state = next_state
        # send action to ant
        return next_state, next_ants, ant_rewards, self.DONE, loc_dict

    def start_server(self, port_num):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", port_num))
        server.listen(2)
        try:
            self.connection, address = server.accept()
            while True:
                received = self.connection.recv(65536)
                if received is not None:
                    data_arr = pickle.loads(received)
                    if data_arr[0] == -2 and data_arr[1] == -1:
                        self.state_queue.empty()
                        # print('got state:')
                        self.state_queue.put(data_arr[2])
                    if data_arr[3] == -1 and data_arr[4] == -2:
                        self.ants_loc_queue.empty()
                        # print('got ant:')
                        self.ants_loc_queue.put(data_arr[5:])
        except Exception as err:
            self.DONE = True
            print('thread %s' % self.Env_name, err)
        finally:
            self.connection.close()

    def generate_ant_reward(self, actions, map_state_next):
        # print("actions:", actions)
        loc_dict = {}
        rewards = {}
        i = 0
        for _ in range(int(len(actions) / 2)):
            reward = -100
            loc = actions[i]
            act = actions[i + 1]
            next_target_loc = get_next_loc(act, loc)
            loc_dict[loc] = loc
            ######################################################################################
            # if map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.MY_ANT\
            #         or map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.HILL:
            #     if loc == next_target_loc:
            #         reward = -1
            #     else:
            #         reward = - 0.5
            #     loc_dict[loc] = next_target_loc
            #     if self.has_eat_food(next_target_loc):
            #         reward = Constants.GET_FOOD_REWARD
            # else:
            #     if map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.DEAD:
            #         reward = Constants.DEAD_ANT_REWARD
            #         loc_dict[loc] = next_target_loc
            #     else:
            #         if map_state_next[loc[0]][loc[1]] == Constants.MY_ANT:
            #             reward = -5
            #             loc_dict[loc] = loc
            #         if map_state_next[loc[0]][loc[1]] == Constants.DEAD:
            #             reward = Constants.DEAD_ANT_REWARD
            #             loc_dict[loc] = loc
            ######################################################################################
            if map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.MY_ANT\
                    or map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.HILL:
                loc_dict[loc] = next_target_loc
                # d = self.distance_to_food(next_target_loc)

                if self.has_eat_food(next_target_loc):
                    reward = Constants.GET_FOOD_REWARD
                else:
                    reward = 0
            else:
                if map_state_next[next_target_loc[0]][next_target_loc[1]] == Constants.DEAD:
                    reward = Constants.DEAD_ANT_REWARD
                    loc_dict[loc] = next_target_loc
                else:
                    if map_state_next[loc[0]][loc[1]] == Constants.MY_ANT:
                        reward = 0
                        loc_dict[loc] = loc
                    if map_state_next[loc[0]][loc[1]] == Constants.DEAD:
                        reward = Constants.DEAD_ANT_REWARD
                        loc_dict[loc] = loc
            ######################################################################################
            if reward == -100:
                print("reward == -100!!!!  target " + str(map_state_next[next_target_loc[0]][next_target_loc[1]]) +
                      " " + str(map_state_next[loc[0]][loc[1]]))
            i += 2
            # print(str(next_target_loc) + " reward = " + str(reward))

            rewards[loc] = reward
        return rewards, loc_dict

    def has_eat_food(self, next_loc):
        x, y = getCorrectCoord(next_loc[0] - 1, next_loc[1])

        if self.state[x][y] == Constants.FOOD:
            return True
        x, y = getCorrectCoord(next_loc[0] + 1, next_loc[1])
        if self.state[x][y] == Constants.FOOD:
            return True
        x, y = getCorrectCoord(next_loc[0], next_loc[1] + 1)
        if self.state[x][y] == Constants.FOOD:
            return True
        x, y = getCorrectCoord(next_loc[0], next_loc[1] - 1)
        if self.state[x][y] == Constants.FOOD:
            return True
        return False

    def distance_to_food(self, next_loc):
        x, y = getCorrectCoord(next_loc[0] - 1, next_loc[1] - 1)
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0] - 1, next_loc[1] + 1)
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0] + 1, next_loc[1] - 1)
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0] + 1, next_loc[1] + 1)
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0] + 2, next_loc[1])
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0] - 2, next_loc[1])
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0], next_loc[1] + 2)
        if self.state[x][y] == Constants.FOOD:
            return 1
        x, y = getCorrectCoord(next_loc[0], next_loc[1] - 2)
        if self.state[x][y] == Constants.FOOD:
            return 1

        x, y = getCorrectCoord(next_loc[0] - 3, next_loc[1])
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] - 2, next_loc[1] + 1)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] - 2, next_loc[1] - 1)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] - 1, next_loc[1] + 2)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] - 1, next_loc[1] -2)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0], next_loc[1] + 3)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0], next_loc[1] - 3)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] + 1, next_loc[1] + 2)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] + 1, next_loc[1] - 2)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] + 2, next_loc[1] + 1)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] + 2, next_loc[1] - 1)
        if self.state[x][y] == Constants.FOOD:
            return 2
        x, y = getCorrectCoord(next_loc[0] + 3, next_loc[1])
        if self.state[x][y] == Constants.FOOD:
            return 2
        return -1


def get_next_loc(act, loc):
    next_loc = (0, 0)
    if act == 0:
        next_loc = loc
    elif act == 1:
        next_loc = (loc[0] - 1, loc[1])
    elif act == 2:
        next_loc = (loc[0], loc[1] + 1)
    elif act == 3:
        next_loc = (loc[0] + 1, loc[1])
    elif act == 4:
        next_loc = (loc[0], loc[1] - 1)
    x_,y_ = getCorrectCoord(next_loc[0], next_loc[1])
    next_loc = (x_, y_)

    # if next_loc[0] < 0:
    #     next_loc = (MAP_HEIGHT + next_loc[0], next_loc[1])
    # elif next_loc[0] >= MAP_HEIGHT:
    #     next_loc = (next_loc[0] - MAP_HEIGHT, next_loc[1])
    # if next_loc[1] < 0:
    #     next_loc = (MAP_WIDTH + next_loc[0], next_loc[1])
    # elif next_loc[1] >= MAP_WIDTH:
    #     next_loc = (next_loc[0] - MAP_WIDTH, next_loc[1])
    return next_loc

def getCorrectCoord(x, y):
    x_ = x
    y_ = y
    if x < 0:
        x_ = Constants.MAP_HEIGHT + x
    elif x >= Constants.MAP_HEIGHT:
        x_ = x - Constants.MAP_HEIGHT
    if y < 0:
        y_ = Constants.MAP_WIDTH + y
    elif y >= Constants.MAP_WIDTH:
        y_ = y - Constants.MAP_WIDTH
    return x_, y_

