import tensorflow as tf
from tf.keras.layers import PReLU
from tflearn.layers.core import fully_connected
import numpy as np
prelu = PReLU()
# simple
input = tf.placeholder(tf.float32, [None, 2])
layer = fully_connected(input, 2, activation='linear', weights_init='xavier')
layer = prelu(layer)
layer = fully_connected(layer, 2, activation='linear', weights_init='xavier')
layer = prelu(layer)
print(prelu.trainable_variables)
config = tf.ConfigProto()
init = tf.global_variables_initializer()

# Launch the session
sess = tf.Session(config=config)
sess.run(init)
print(sess.run((layer), feed_dict={input: np.array([2,2]).reshape(1,2)}))
