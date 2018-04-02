import multiprocessing as mp
import EnvTest
import threading


def choose_action():
    return 's'


class Worker(object):
    def __init__(self, name, index):
        self.task_name = name
        self.task_index = index
        self.env = EnvTest.AntEnv(name)

    def work(self):
        print('Start Worker: ', self.task_index)
        state_c = self.env.reset()  # could be abandon?
        print("worker", self.task_index, state_c)

        ep_r = 0

        while True:
            state_queue_in_one_step = []
            ants_done = False
            while not ants_done:
                s_a, loc = self.env.get_ant_State()
                state_queue_in_one_step.append((s_a, loc))
                action = choose_action(s_a)
                ants_done = self.env.take_action(action, loc)

            reward, Done = self.env.getReward()



if __name__ == "__main__":
    workers = []
    # Create Worker
    for i in range(2):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, i))

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
