import numpy as np
import pickle
import os
import socket
from queue import Queue
import threading

Start_play_command = 'D:\Python27\python tools/playgame.py "python %s" "python tools/sample_bots/python/HunterBot.py"  ' \
                     '--map_file "tools/maps/example/tutorial1.map" --log_dir %s --turns 60 --scenario   --nolaunch' \
                     ' --player_seed 7  --turntime 5000 -e'
# --verbose   --nolaunch

PORT1 = 28029
PORT2 = 28040
PORT3 = 28038


class AntEnv:
    def __init__(self, name):
        self.Env_name = name
        self.observation_space_shape = 1677 + 2  # 43*39=1677
        self.action_space_num = 5  # Action space  # 0--stay 1--North 2--East 3--South 4--West
        cols = 20
        rows = 20
        self.stepNum = 0
        self.DONE = True
        self.state = [0 for _ in range(cols*rows)]
        self.s1 = np.array(self.state)
        self.ants_loc_queue = Queue()
        self.state_queue = Queue()
        self.connection = None

    def reset(self):
        command = ''
        self.DONE = False
        self.state_queue.empty()
        self.ants_loc_queue.empty()
        if self.Env_name == 'W_0':
            command = Start_play_command % ('MyBot_1.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT1,))
            t.start()
            print(command)
        elif self.Env_name == 'W_1':
            command = Start_play_command % ('MyBot_2.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT2,))
            t.start()
            print(command)
        os.popen(command)
        tmp_state = self.state_queue.get(timeout=300)
        tmp_state = np.array(tmp_state)
        tmp_ants = self.ants_loc_queue.get(timeout=300)
        return tmp_state, tmp_ants

    def step(self, actions):

        next_state = None
        next_ants = None
        print("action = ", actions)
        self.connection.sendall(pickle.dumps(actions, protocol=2))

        try:
            next_state = self.state_queue.get(block=True, timeout=5)
            next_state = np.array(next_state)
            next_ants = self.ants_loc_queue.get(block=True, timeout=2)

        except Exception as err:
            self.DONE = True
        print("next_ants = ", next_ants)
        # reward = len(next_ants)
        # send action to ant
        return next_state, next_ants, 10, self.DONE

    def start_server(self, port_num):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server.bind(("127.0.0.1", port_num))
        server.listen(2)
        try:
            self.connection, address = server.accept()
            while True:
                received = self.connection.recv(32768)
                if received is not None:
                    data_arr = pickle.loads(received)
                    if data_arr[0] == -2 and data_arr[1] == -1:
                        self.state_queue.empty()
                        # print('got state:', data_arr)
                        self.state_queue.put(data_arr[2:])
                    if data_arr[0] == -1 and data_arr[1] == -2:
                        self.ants_loc_queue.empty()
                        # print('got ant:', data_arr)
                        self.ants_loc_queue.put(data_arr[2:])
        except Exception as err:
            self.DONE = True
            print('lk', err)
        finally:
            self.connection.close()

    def step_for_ant(self, action, loc):
        output = [0, loc, action]
        test = bytes(output)
        print(int(test[0]))
        try:
            self.connection.sendall(bytes(output))
        except Exception as err:
            print(err)
