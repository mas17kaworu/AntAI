import multiprocessing as mp
import EnvTest
import threading


def choose_action(state):
    return b'w'


class Worker(object):
    def __init__(self, name, index):
        self.task_name = name
        self.task_index = index
        self.env = EnvTest.AntEnv(name)

    def work(self):
        def get_ant_state(map, loc):
            return [1, 2]  # only for test

        print('Start Worker: ', self.task_index)

        for _ in range(3):
            state_map = self.env.reset()
            print("worker", self.task_index, "receive first state", state_map)
            steps_num = 1
            ep_r = 0
            Done = False
            while not Done:
                # print('start a new step')
                for index in range(state_map[0]):
                    # get state for each ant
                    loc = (0, 0)  # only for test
                    s_a = get_ant_state(state_map, loc)
                    # get action for each ant
                    action = choose_action(s_a)
                    self.env.step_for_ant(action)
                state_map_, reward, Done = self.env.step()
                steps_num += 1
                print(steps_num)
                # do update N-Network
                state_map = state_map_

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
