dataet=[[[3, 5], [7, 2], [7, 6]],
        [[2, 5], [1, 3], [4, 3]],
        [[8, 1], [1, 8], [9, 3]],
        [[1, 5], [6, 7], [4, 9]]]



import tensorflow as tf
import numpy as np


inut=tf.placeholder(dtype=tf.int32,shape=[4,3,2])

data=tf.unstack(inut,3,1)

with tf.Session() as sess:
    print(sess.run(data,feed_dict={inut:dataet}))
