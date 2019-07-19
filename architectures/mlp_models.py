'''
define the mlp model for the mpnet in the baxter environment
'''
from . mlp_pipeline import mlp_pipeline
import tensorflow as tf
from tensorflow.python.keras.layers import PReLU
def wrap_prelu(in_signal):
    # first time call this will construct class
    # tensorflow will use this constructed PReLU afterwards without constructing again
    prelu = PReLU()
    return prelu(in_signal, channel_shared=True)
def baxter_mpnet_mlp(input_size, output_size):
    mlp = mlp_pipeline

    mlp_arguments = {'layer_sizes': [1280, 896, 512, 384, 256, 128, 64, 32, output_size],
                     'non_linearity': wrap_prelu,
                     'regularizer': None,
                     'weight_decay': 0.0,
                     'reuse': False, 'scope': None, 'dropout_prob': 0.5, 'verbose': True
                    }
    return mlp, mlp_arguments
