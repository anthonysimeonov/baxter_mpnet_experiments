import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

# tensorflow code for MLP
def mlp_pipeline(in_signal, layer_sizes=[], non_linearity=tf.nn.relu, regularizer=None,
                        weight_decay=0., reuse=False, scope=None, dropout_prob=None,
                        verbose=False):
    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers([dropout_prob], n_layers)
    layer = in_signal
    for i in xrange(0, n_layers - 2):
        name = 'mlp_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='uniform_scaling', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if non_linearity is not None:
            layer = non_linearity(layer, name='alpha_%d' % (i))

        if dropout_prob is not None and dropout_prob[i] > 0:
            print('before dropout:')
            print(layer)
            layer = tf.nn.dropout(layer, 1.0 - dropout_prob[i])
            #layer = dropout(layer, 1.0 - dropout_prob[i])  ### this is not working
            print('after dropout:')
            print(layer)

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last 2nd fc layer never has a dropout layer, but has a non-linearity.
    name = 'mlp_fc_' + str(n_layers - 2)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 2], activation='linear', weights_init='uniform_scaling', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if non_linearity is not None:
        layer = non_linearity(layer, name='alpha_%d' % (n_layers-2))

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last fc layer never has a non-linearity.
    name = 'mlp_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='uniform_scaling', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer
