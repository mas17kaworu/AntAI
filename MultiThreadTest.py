import numpy as np
import tensorflow as tf
import EnvTest
import threading

GLOBAL_NET_SCOPE = 'global_net'
UPDATE_GLOBAL_ITER = 1
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

MAX_GLOBAL_EP = 1000
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
THREAD_NUM = 4

env = EnvTest.AntEnv("-1")
N_S = env.observation_space_shape
N_A = env.action_space_num


class ACNet(object):
    def __init__(self, scope, global_net=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')  # Action history
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                        axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            self.global_step = tf.train.get_or_create_global_step()
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, global_net.a_params), global_step=self.global_step)
                    self.update_c_op = opt_c.apply_gradients(zip(self.c_grads, global_net.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 5000, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 5000, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        # print("shape", prob_weights.shape[1])
        # print("prob", prob_weights.ravel())
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])


def choose_action_only_for_test(state):
    action = np.random.randint(0, 5)
    return action  # 0--stay 1--North 2--East 3--South 4--West


class Worker(object):
    def __init__(self, name, index, globalAC):
        self.task_name = name
        self.task_index = index
        self.env = EnvTest.AntEnv(name)
        self.AC = ACNet(name, globalAC)

    def work(self):
        def get_ant_state(map_, loc_):
            map_extend = np.insert(map_, map_.shape, loc_)
            return map_extend  # only for test

        global GLOBAL_RUNNING_R, GLOBAL_EP
        print('Start Worker: ', self.task_index)
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        for _ in range(1):  # while not(COORD.should_stop()) and (GLOBAL_EP < MAX_GLOBAL_EP):
            state_map, ants_loc = self.env.reset()
            state_map = state_map.flatten()
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
                    action = self.AC.choose_action(s_a)
                    # action = choose_action_only_for_test(s_a)

                    action_nomal = action.tolist()
                    actions_queue.append(loc)
                    actions_queue.append(action_nomal)

                    buffer_a.append(action)
                    buffer_s.append(s_a)
                state_map_, ants_loc_, reward, Done = self.env.step(actions_queue)
                state_map_ = state_map_.flatten()
                ep_r += reward

                buffer_r.append(reward)
                # do update N-Network
                if total_step % UPDATE_GLOBAL_ITER == 0 or Done:
                    if Done:
                        v_s_ = 0
                    else:
                        extend_s_m_ = np.insert(state_map_, state_map_.shape, (0, 0))
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: extend_s_m_[np.newaxis, :]})[0, 0]  # RNN? use s_a in one step
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                actions_queue.clear()
                steps_num += 1
                # print(steps_num)
                state_map = state_map_
                ants_loc = ants_loc_

            print('episode : finished')



if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        opt_a = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        opt_c = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        global_net = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        # Create Worker
        for i in range(1):
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, i, global_net))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
