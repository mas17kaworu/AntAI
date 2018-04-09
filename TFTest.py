import tensorflow as tf
import numpy as np

# z = np.random.randint(0, 10, size=[5])
z = tf.constant([4, 0, 0, 0])
x = tf.constant([0.5, 0.2, 0.3, 0, 0])
y = tf.one_hot(z, 5, dtype=tf.float32)
aa = x * y
log_prob = tf.reduce_sum(aa, axis=1, keep_dims=True)
with tf.Session()as sess:
    print(sess.run(z))
    print(sess.run(y))
    print(sess.run(aa))
    print(sess.run(log_prob))
