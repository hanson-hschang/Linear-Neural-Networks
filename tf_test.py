import tensorflow as tf
import numpy as np

print("version of TensorFlow = ", tf.__version__)

a = tf.constant([[1,2],[1,3],[1,4]])
b = tf.constant([[2,3],[2,4],[2,5]])

print(a.get_shape())
print(b.get_shape())

c = tf.tensordot(a,b,[0,0])
ab = tf.reduce_sum(c)

a = tf.placeholder(tf.float32, shape=(5))
b = tf.placeholder(tf.float32, shape=(5))

dot_a_b = tf.tensordot(a, b, 1)

with tf.Session() as sess:
    print(dot_a_b.eval(feed_dict={a: [1, 2, 3, 4, 5], b: [6, 7, 8, 9, 10]}))

# with tf.Session() as sess:
#     print(a.eval())
#     print(b.eval())
#     print(c.eval())
#     print(ab.eval())