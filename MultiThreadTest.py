import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import EnvTest
import threading
import antLog
import Constants
import time

GLOBAL_NET_SCOPE = 'global_net'
UPDATE_GLOBAL_ITER = 30

GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic

MAX_GLOBAL_EP = 5000
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
THREAD_NUM = 6

load_model = False
SAVE_PER_EPISODE = 100
SHOULD_SAVE = True

env = EnvTest.AntEnv("-1")
N_S = env.observation_space_shape
N_A = env.action_space_num

SMALL_MAP_WIDTH = 5
SMALL_MAP_HEIGHT = 5
N_S_ACTOR = SMALL_MAP_WIDTH * SMALL_MAP_HEIGHT


class ACNet(object):
    def __init__(self, scope, global_net=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S_ACTOR], 'S')
                # self.s_actor = tf.placeholder(tf.float32, [None, N_S_ACTOR], 'S_Actor')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S_ACTOR], 'S')
                # self.s_actor = tf.placeholder(tf.float32, [None, N_S_ACTOR], 'S_Actor')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')  # Action history
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                self.td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(self.td))
                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                        axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(self.td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v

                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    # self.a_loss = tf.clip_by_value(tf.reduce_mean(-self.exp_v), -0.5, 0.5)  # clip a loss

                # with tf.name_scope('choose_A'):
                #     self.A = np.random.choice(range(self.a_prob.shape[1]), p=self.a_prob.ravel())
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
        w_init = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope('critic'):  # only critic controls the cnn update
            image_in = tf.reshape(self.s, [-1, SMALL_MAP_WIDTH, SMALL_MAP_HEIGHT, 1])
            conv_1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=image_in,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding="VALID")
            # conv_2 = slim.conv2d(activation_fn=tf.nn.elu,
            #                      inputs=conv_1,
            #                      num_outputs=32,
            #                      kernel_size=[3, 3],
            #                      stride=[1, 1])
            after_cnn = slim.flatten(conv_1)
            print(after_cnn)
            hidden = slim.fully_connected(slim.flatten(conv_1), 512, activation_fn=tf.nn.relu)

            # RNN Cell
            cell_size = 256
            rnn_input = tf.expand_dims(hidden, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=rnn_input, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation

            l_c = tf.layers.dense(cell_out, 256, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            # l_a = tf.layers.dense(self.s_actor, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # l_a2 = tf.layers.dense(l_a, 200, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            # a_prob = tf.layers.dense(l_a2, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

            # l_c = tf.layers.dense(self.s, 2000, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            # l_c2 = tf.layers.dense(l_c, 2000, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            # v = tf.layers.dense(l_c2, 1, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(cell_out, 128, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = slim.fully_connected(l_a, N_A, activation_fn=tf.nn.softmax)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s, cell_state, task_index):  # run by a local
        prob_weights, cell_state = SESS.run([self.a_prob, self.final_state],
                                            feed_dict={self.s: s[np.newaxis, :], self.init_state: cell_state})
        # print("s shape = ", s.shape)
        if task_index == 0:
            antLog.write_log(str(s.reshape([SMALL_MAP_WIDTH, SMALL_MAP_HEIGHT])), "Prob")
            antLog.write_log('prob = ' + str(prob_weights), "Prob")
        # print("prob", prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, cell_state

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
        self.ants = []

    def work(self, saver):
        def get_ant_state(map_, loc_):
            # print(map_.shape)
            small_map = np.zeros((SMALL_MAP_HEIGHT, SMALL_MAP_WIDTH))
            x = loc_[0] - int(SMALL_MAP_WIDTH / 2)
            for n in range(SMALL_MAP_HEIGHT):
                if x < 0:
                    x = x + Constants.MAP_HEIGHT
                elif x >= Constants.MAP_HEIGHT:
                    x = x - Constants.MAP_HEIGHT
                y = loc_[1] - int(SMALL_MAP_HEIGHT / 2)
                for m in range(SMALL_MAP_WIDTH):
                    if y < 0:
                        y = y + Constants.MAP_WIDTH
                    elif y >= Constants.MAP_WIDTH:
                        y = y - Constants.MAP_WIDTH
                    # print('x=',x ,' y=',y)
                    small_map[n][m] = map_[x][y]
                    y += 1
                x += 1
            # antLog.write_log('small_map = ' + str(small_map), "Task0Summary")
            small_map = small_map.flatten()
            return small_map

        global GLOBAL_RUNNING_R, GLOBAL_EP
        print('Start Worker: ', self.task_index)
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
        # for _ in range(1):
        while not(COORD.should_stop()) and (GLOBAL_EP < MAX_GLOBAL_EP):
            state_map, ants_loc, game_done = self.env.reset()
            # print("worker", self.task_index, "receive first state", state_map)
            steps_num = 1
            ants_steps = 1
            max_ants_num = 0
            ep_r = 0
            if not game_done:
                next_loc = ants_loc[0]
            loc_last = None
            actions_queue = []
            rnn_state = SESS.run(self.AC.init_state)
            # self.batch_rnn_state = rnn_state
            keep_state = rnn_state.copy()
            # self.ants = ants_loc
            while not game_done:
                # print('start a new step')
                # print("ants_loc", ants_loc)
                # print("next_loc = ", next_loc)
                flag_not_dead = False

                for loc in ants_loc:
                    if loc == next_loc:  # only trace first ant
                        # print("loc = nextloc", loc)
                        # get state for each ant
                        s_a = get_ant_state(state_map, loc)
                        # get action for each ant
                        action, rnn_state_ = self.AC.choose_action(s_a, rnn_state, self.task_index)

                        action_normal = action.tolist()
                        loc_last = loc
                        buffer_a.append(action)
                        buffer_s.append(s_a)

                        actions_queue.append(loc)
                        actions_queue.append(action_normal + 1)
                        flag_not_dead = True
                    else:
                        actions_queue.append(loc)
                        actions_queue.append(0)  # don't move

                # print("actions ", actions_queue)
                if flag_not_dead:
                    state_map_, ants_loc_, rewards, game_done, loc_dict = self.env.step(actions_queue)
                else:
                    rewards = [0]
                    game_done = True
                    print("ant dead")
                    time.sleep(3.5)
                if max_ants_num < len(rewards):
                    max_ants_num = len(rewards)
                # print("ants_loc", ants_loc)
                # print("ants_loc_", ants_loc_)
                # print("loc_dict", loc_dict)
                if self.task_index == 0:
                    antLog.write_log('GLOBAL_EP = ' + str(GLOBAL_EP), "Task0Summary")
                    antLog.write_log('step = ' + str(steps_num), "Task0Summary")
                    antLog.write_log('actions = ' + str(actions_queue), "Task0Summary")
                    antLog.write_log('rewards = ' + str(rewards), "Task0Summary")
                    antLog.write_log('new_ants_loc_ = ' + str(ants_loc_), "Task0Summary")
                    # antLog.write_log('reward= ' + str(rewards), "Task0Summary")
                if not game_done:
                    next_loc = loc_dict[loc_last]
                    for loc in ants_loc:
                        if loc == loc_last:
                            buffer_r.append(rewards[loc])  # choose the right reward, then put it into queue
                            ep_r += rewards[loc]
                            break  # only trace first ant
                else:
                    buffer_r.append(0)
                # print("buffer_r" + str(buffer_r))
                # do update N-Network
                if steps_num % UPDATE_GLOBAL_ITER == 0 or buffer_r[-1] == Constants.DEAD_ANT_REWARD or game_done:
                    ants_steps = 1
                    buffer_s_next = []
                    s_a_ = None
                    # print("buffer_r" + str(buffer_r))
                    if game_done or buffer_r[-1] == Constants.DEAD_ANT_REWARD:
                        v_s_ = 0  # todo game
                    else:
                        for loc in ants_loc:  # some Ants has dead
                            if loc == loc_last:
                                s_a_ = get_ant_state(state_map_, next_loc)
                                buffer_s_next.append(s_a_)
                                # print("s_a_" + str(s_a_))
                                break  #
                        # print(str(buffer_s_next))
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_a_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]  # RNN? use s_a in one step
                        # print("value from net ", v_s_)

                    buffer_v_target = []
                    # print("value from net ", v_s_)
                    if not game_done:
                        for r in buffer_r[::-1]:
                            if r == Constants.DEAD:
                                v_s_ = r
                            else:
                                v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()
                        # print("buffer_v_target" + str(buffer_v_target))
                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                        feed_dict = {
                            self.AC.s: buffer_s,
                            self.AC.a_his: buffer_a,
                            self.AC.v_target: buffer_v_target,
                            self.AC.init_state: keep_state,
                        }
                        if self.task_index == 0 and steps_num % 1 == 0:
                            loss_a = SESS.run(self.AC.a_loss, feed_dict)
                            loss_c = SESS.run(self.AC.c_loss, feed_dict)
                            v_record = SESS.run(self.AC.v, feed_dict)

                            antLog.write_log('rewards = ' + str(buffer_r), "Task0Summary")
                            antLog.write_log("v = " + str(v_record), "Task0Summary")
                            antLog.write_log("v_s_ = " + str(buffer_v_target), "Task0Summary")
                            antLog.write_log("loss_a = %f, loss_c = %f" % (loss_a, loss_c), "Task0Summary")
                            # antLog.write_log('td =' + str(td), "Task0Summary")

                        self.AC.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
                        self.AC.pull_global()
                        keep_state = rnn_state_.copy()
                    elif self.task_index == 0:
                        antLog.write_log("Game Done", "Task0Summary")
                # if not game_done:
                #     if loc_last not in ants_loc_:
                #         next_loc = ants_loc_[0]

                actions_queue.clear()
                steps_num += 1
                ants_steps += 1
                # print(steps_num)
                state_map = state_map_
                ants_loc = ants_loc_

                if game_done:
                    GLOBAL_EP += 1
                    buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
                    antLog.write_log('ep_r = ' + str(ep_r), "Total")
                    print("worker ", self.task_name, " max_ants_num = ", max_ants_num, "er_r = ", ep_r
                          , "GLOBAL_EP = ", GLOBAL_EP)
                    max_ants_num = 0

            if GLOBAL_EP % SAVE_PER_EPISODE == 0 and SHOULD_SAVE:
                saver.save(SESS, model_path + '/model-' + str(GLOBAL_EP) + '.cptk')
                print("Saved Model")

            print('episode : finished')



model_path = './model'

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):

        opt_a = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')  #RMSPropOptimizer
        opt_c = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        global_net = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        # Create Worker
        for i in range(THREAD_NUM):
            i_name = 'W_%i' % (i + 1)
            workers.append(Worker(i_name, i, global_net))

        saver = tf.train.Saver(max_to_keep=1)

    COORD = tf.train.Coordinator()

    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(SESS, ckpt.model_checkpoint_path)
    else:
        SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(saver)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)


def get_next_loc(act, loc):
    next_loc = (0, 0)
    if act == 0:
        next_loc = loc
    elif act == 1:
        next_loc = (loc[0] - 1, loc[1])
    elif act == 2:
        next_loc = (loc[0], loc[1] + 1)
    elif act == 3:
        next_loc = (loc[0] + 1, loc[1])
    elif act == 4:
        next_loc = (loc[0], loc[1] - 1)
    x_,y_ = getCorrectCoord(next_loc[0], next_loc[1])
    next_loc = (x_, y_)
    return next_loc

def getCorrectCoord(x, y):
    x_ = x
    y_ = y
    if x < 0:
        x_ = Constants.MAP_HEIGHT + x
    elif x >= Constants.MAP_HEIGHT:
        x_ = x - Constants.MAP_HEIGHT
    if y < 0:
        y_ = Constants.MAP_WIDTH + y
    elif y >= Constants.MAP_WIDTH:
        y_ = y - Constants.MAP_WIDTH
    return x_, y_