'''
define the mlp model for the mpnet in the baxter environment
'''
from . mlp_pipeline import mlp_pipeline
import tensorflow as tf
import tensorflow as tf
def wrap_prelu(_x, name=None):
    # ref: https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    alphas = tf.get_variable(name, [1],
                           initializer=tf.constant_initializer(0.25),
                            dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg
def baxter_mpnet_mlp(input_size, output_size):
    mlp = mlp_pipeline

    mlp_arguments = {'layer_sizes': [1280, 896, 512, 384, 256, 128, 64, 32, output_size],
                     'non_linearity': tf.nn.relu,
                     'regularizer': None,
                     'weight_decay': 0.0,
                     'reuse': False, 'scope': None, 'dropout_prob': None, 'verbose': True
                    }
    return mlp, mlp_arguments
