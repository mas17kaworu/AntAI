import multiprocessing as mp
import EnvTest
import threading


class Worker(object):
    def __init__(self, name, index):
        self.task_name = name
        self.task_index = index
        self.env = EnvTest.AntEnv(name)

    def work(self):
        print('Start Worker: ', self.task_index)
        state = self.env.reset()
        print("worker", self.task_index, state)






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
