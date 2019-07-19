'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only, linear_encoder
from tensorflow.python.keras.layers import PReLU
def wrap_prelu(in_signal):
    # first time call this will construct class
    # tensorflow will use this constructed PReLU afterwards without constructing again
    prelu = PReLU()
    return prelu(in_signal, channel_shared=True)

def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def linear_ae(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = linear_encoder
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'layer_sizes': [786, 512, 256, bneck_size],
                    'verbose': True,
                    'weight_decay': 0.,
                    'non_linearity': wrap_prelu
                    }

    decoder_args = {'layer_sizes': [256, 512, 786, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True,
                    'weight_decay': 0.,
                    'non_linearity': wrap_prelu
                    }

    return encoder, decoder, encoder_args, decoder_args



def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 500,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return params
