import multiprocessing as mp
import numpy as np
import EnvTest
import threading


def choose_action(state):
    action = np.random.randint(0, 5)
    return action  # 0--stay 1--North 2--East 3--South 4--West


class Worker(object):
    def __init__(self, name, index):
        self.task_name = name
        self.task_index = index
        self.env = EnvTest.AntEnv(name)

    def work(self):
        def get_ant_state(map, loc):
            return [1, 2]  # only for test

        print('Start Worker: ', self.task_index)

        for _ in range(1):
            state_map, ants_loc = self.env.reset()
            # print("worker", self.task_index, "receive first state", state_map)
            steps_num = 1
            ep_r = 0
            Done = False
            actions_queue = []
            while not Done:
                # print('start a new step')
                for loc in ants_loc:
                    # get state for each ant
                    s_a = get_ant_state(state_map, loc)
                    # get action for each ant
                    action = choose_action(s_a)
                    actions_queue.append(loc)
                    actions_queue.append(action)
                state_map_, ants_loc_, reward, Done = self.env.step(actions_queue)
                actions_queue.clear()
                steps_num += 1
                # print(steps_num)
                # do update N-Network
                state_map = state_map_
                ants_loc = ants_loc_

            print('episode : finished')



if __name__ == "__main__":
    workers = []
    # Create Worker
    for i in range(1):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, i))

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
