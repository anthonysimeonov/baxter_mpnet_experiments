'''
define the mlp model for the mpnet in the baxter environment
'''
from . mlp_pipeline import mlp_pipeline
import tensorflow as tf
def baxter_mpnet_mlp(input_size, output_size):
    mlp = mlp_pipeline
    mlp_arguments = {'layer_sizes': [1280, 896, 512, 384, 256, 128, 64, 32, output_size],
                     'non_linearity': tf.nn.relu,
                     'regularizer': None,
                     'weight_decay': 0.001,
                     'reuse': False, 'scope': None, 'dropout_prob': 0.5, 'verbose': True
                    }
    return mlp, mlp_arguments
