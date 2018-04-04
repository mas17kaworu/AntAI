import numpy as np
import subprocess
import multiprocessing
import pickle
import os
import socket
from queue import Queue
import threading

Start_play_command = 'C:\Python27\python tools/playgame.py "python %s" "python tools/sample_bots/python/HunterBot.py"  ' \
                     '--map_file "tools/maps/example/tutorial1.map" --log_dir %s --turns 60 --scenario --nolaunch  ' \
                     '--player_seed 7   -e'
# --verbose   --nolaunch

PREFIX_S = b'state:'
PORT1 = 28029
PORT2 = 28040
PORT3 = 28038


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

        self.queue = Queue()
        self.connection = None

    def reset(self):
        command = ''
        self.DONE = False

        if self.Env_name == 'W_0':
            command = Start_play_command % ('MyBot_1.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT1, self.queue))
            t.start()
            print(command)
        elif self.Env_name == 'W_1':
            command = Start_play_command % ('MyBot_2.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=self.start_server, args=(PORT2, self.queue))
            t.start()
            print(command)
###########################################################################################
        # command = 'python stdCommTest.py'
        # p = subprocess.run(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        #
        # p.stdin.write(b"Hello\n")
        # p.stdin.flush()
        # print('got', p.stdout.readline().strip())
        # p.stdin.write(b"How are you?\n")
        # p.stdin.flush()
        # print('got', p.stdout.readline().strip())
#############################################################################################
        os.popen(command)
        return self.queue.get(timeout=2000)

    def step(self):
        reward = 0  # only for test
        next_state = None
        try:
            next_state = self.queue.get(block=True, timeout=5)
        except Exception as err:
            self.DONE = True

        # send action to ant
        return next_state, reward, self.DONE

    def start_server(self, portNum, queue):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server.bind(("127.0.0.1", portNum))
        server.listen(1)
        # command = 'python socketCommTest.py'
        # os.system(command)
        try:
            self.connection, address = server.accept()
            while True:
                received = self.connection.recv(1024)
                if received != None:
                    data_arr = pickle.loads(received)
                    if data_arr[0] == -2 and data_arr[1] == -1:
                        print('got state:', data_arr, ' Ants num = ', data_arr[0], 'first element', data_arr[1])
                        queue.put(data_arr)
                    if data_arr[0] == -1 and data_arr[1] == -2:
                        print('got ant:', data_arr, ' Ants num = ', data_arr[0], 'first element', data_arr[1])
                        # queue.put(data_arr)
                    # output = "test: %s" % received
                    # connection.sendall(output.encode('utf-8'))
        except Exception as err:
            self.DONE = True
            print('lk', err)
        finally:
            self.connection.close()

    def step_for_ant(self, action, loc):
        output = [loc[0], loc[1], action]
        try:
            self.connection.sendall(pickle.dumps(output))
        except Exception as err:
            print(err)
