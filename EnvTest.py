import numpy as np
import subprocess
import multiprocessing
import os
import socket
from queue import Queue
import threading

Start_play_command = 'D:\Python27\python tools/playgame.py "python %s" "python tools/sample_bots/python/HunterBot.py"  ' \
                     '--map_file "tools/maps/example/tutorial1.map" --log_dir %s --turns 60 --scenario  ' \
                     '--player_seed 7 --nolaunch  -e'

#--verbose


PORT1 = 8039
PORT2 = 8040
PORT3 = 8038


def start_server(portNum, queue):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.bind(("127.0.0.1", portNum))
    server.listen(1)
    # command = 'python socketCommTest.py'
    # os.system(command)
    try:
        connection, address = server.accept()
        while True:
            received = connection.recv(1024)
            if received != None:
                print('got:', received)
                queue.put(received)
                connection.send(b"test: %s" % received)
    except Exception as err:
        print(err)
    finally:
        connection.close()


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

    def reset(self):
        command = ''
        self.DONE = False

        if self.Env_name == 'W_0':
            command = Start_play_command % ('MyBot_1.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=start_server, args=(PORT1, self.queue))
            t.start()
            # threading._start_new_thread(start_server, (PORT1,))
            print(command)
        elif self.Env_name == 'W_1':
            command = Start_play_command % ('MyBot_2.py', ('ant_log_' + self.Env_name))
            t = threading.Thread(target=start_server, args=(PORT2, self.queue))
            t.start()
            # threading._start_new_thread(start_server, (PORT2,))
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
        return self.queue.get()

    def step(self, action):
        reward = 0
        # if action == 0:
        #     reward = 10
        # self.stepNum += 1
        # if self.stepNum > 100:
        #     self.DONE = True

        # send action to ant


        return self.s1, reward, self.DONE, 'xxx'

    def one_ant_action(self, action):

        pass
