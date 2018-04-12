#!/usr/bin/env python
from ants import *
import socket
import pickle
import Queue
import threading
import antLog


port_remote = 22038
botMark = 'W_3'

# define a class with a do_turn method
# the Ants.run method will parse and update bot input
# it will also run the do_turn method for us
class MyBot:
    def __init__(self):
        self.test_num = 1
        self.client = None
        self.queue = Queue.Queue()
        # self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # define class level variables, will be remembered between turns
        pass

    # do_setup is run once at the start of the game
    # after the bot has received the game settings
    # the ants class is created and setup by the Ants.run method
    def do_setup(self, ants):
        print("In mybot1 setup")
        antLog.write_log("*************************************setup**************************************", botMark)
        t = threading.Thread(target=self.start_client, args=(port_remote,))
        t.start()
        # self.client.connect(("127.0.0.1", port))

    # do turn is run once per turn
    # the ants class has the game state and is updated by the Ants.run method
    # it also has several helper methods to use
    def do_turn(self, ants):
        # loop through all my ants and try to give them orders
        # the ant_loc is an ant location tuple in (row, col) form
        antLog.write_log("turn %d" % self.test_num, botMark)
        self.queue.empty()
        # generate state for entire map
        map_state = [-2, -1]  # only for test
        map_state.append(ants.map)
        self.client.sendall(pickle.dumps(map_state))

        # send Ants loc to Network
        ants_loc = [-1, -2]
        for ant_loc in ants.my_ants():
            ants_loc.append(ant_loc)
        self.client.sendall(pickle.dumps(ants_loc))

        #  wait for actions
        self.test_num += 1
        data_arr = self.queue.get(timeout=5)
        antLog.write_log(str(data_arr), botMark)
        # waite to receive action
        for index in range(len(ants.my_ants())):

            ant_loc = data_arr[index*2]
            action = choose_action(data_arr[index*2+1])
            if action is None:
                continue
            new_loc = ants.destination(ant_loc, action)
            if ants.passable(new_loc):
                # an order is the location of a current ant and a direction
                ants.issue_order((ant_loc, action))

            # check if we still have time left to calculate more orders
            if ants.time_remaining() < 10:
                break

    def start_client(self, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(("127.0.0.1", port))
        try:
            while True:
                received = self.client.recv(2048)
                data_arr = pickle.loads(received)
                self.queue.put(data_arr)
        except Exception as err:
            pass


def choose_action(number):
    return {
        0: None,
        1: 'n',
        2: 'e',
        3: 's',
        4: 'w'
    }.get(number)


if __name__ == '__main__':
    # psyco will speed up python a little, but is not needed
    try:
        import psyco

        psyco.full()
    except ImportError:
        pass

    try:
        # if run is passed a class with a do_turn method, it will do the work
        # this is not needed, in which case you will need to write your own
        # parsing function and your own game state class
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
