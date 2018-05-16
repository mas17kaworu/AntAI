import tensorflow as tf
import numpy as np

# z = np.random.randint(0, 10, size=[5])
z = tf.constant([4, 0, 0, 0])   # action sequence
x = tf.constant([[0.5, 0.2, 0.1, 0.1, 0.1],     # action posibility for each state
                 [0.1, 0.2, 0.3, 0.4, 0.],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.5, 0.2, 0.3, 0., 0.]])
y = tf.one_hot(z, 5, dtype=tf.float32)
aa = x * y
log_prob = tf.reduce_sum(aa, axis=1, keep_dims=True)        # P(a|s)

v_target = tf.constant([10])
v = tf.constant([11, 9, 9])
td = tf.subtract(v_target, v, name='TD_error')


a_prob = tf.constant([[0.2, 0.3, 0.45, 0.05],
                     [0.25, 0.25, 0.25, 0.25],
                      [0.9, 0.05, 0.05, 0],
                      [0.1, 0.01, 0.01, 0.88]])
tmp0 = tf.log(a_prob + 1e-6)
tmp1 = a_prob * tmp0
entropy = -tf.reduce_sum(tmp1, axis=1, keepdims=True)
with tf.Session()as sess:
    print(sess.run(z))
    print(sess.run(y))
    print(sess.run(aa))
    print(sess.run(log_prob))
    print(sess.run(td))
    print(sess.run(tmp0))
    print(sess.run(tmp1))
    print(sess.run(entropy))
