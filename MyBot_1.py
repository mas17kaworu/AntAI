#!/usr/bin/env python
from ants import *
import sys
import socket

port = 8039
message_=b'bot1'


# define a class with a do_turn method
# the Ants.run method will parse and update bot input
# it will also run the do_turn method for us
class MyBot:
    def __init__(self):
        self.test_num = 1
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # define class level variables, will be remembered between turns
        pass

    # do_setup is run once at the start of the game
    # after the bot has received the game settings
    # the ants class is created and setup by the Ants.run method
    def do_setup(self, ants):
        print("In mybot1 setup")
        ###############################################
        # data = sys.stdin.readline()
        # sys.stdout.write(b"Hm.\n")
        # sys.stdout.flush()
        # data = sys.stdin.readline()
        # sys.stdout.write(b"Whatever.\n")
        # sys.stdout.flush()
        ###############################################
        self.client.connect(("127.0.0.1", port))
        instr = b'setup client'
        self.client.send(instr)
        print(self.client.recv(1024))

    # do turn is run once per turn
    # the ants class has the game state and is updated by the Ants.run method
    # it also has several helper methods to use
    def do_turn(self, ants):
        # loop through all my ants and try to give them orders
        # the ant_loc is an ant location tuple in (row, col) form
        self.test_num += 1
        if self.test_num <= 10:
            self.client.send(b'do_turn_1')
        if self.test_num >= 10:
            self.client.close()
        for ant_loc in ants.my_ants():
            # generate state for each ant
            self.client.send(b'state')
            # waite to receive action
            action = self.client.recv(1024)
            action = 's'  # only for test
            new_loc = ants.destination(ant_loc, action)
            if ants.passable(new_loc):
                # an order is the location of a current ant and a direction
                ants.issue_order((ant_loc, action))

            # check if we still have time left to calculate more orders
            if ants.time_remaining() < 10:
                break




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