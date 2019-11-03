from __future__ import absolute_import
import random
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf


class FractionalPooling2D(Layer):
    def __init__(self, pool_ratio=None, pseudo_random=True, overlap=False, name='FractionPooling2D', **kwargs):
        self.pool_ratio = pool_ratio
        self.input_spec = [InputSpec(ndim=4)]
        self.pseudo_random = pseudo_random
        self.overlap = overlap
        super(FractionalPooling2D, self).__init__(**kwargs)

    def call(self, input):
        [batch_tensor, row_pooling, col_pooling] = tf.nn.fractional_max_pool(
            input, pooling_ratio=self.pool_ratio, pseudo_random=self.pseudo_random, overlapping=self.overlap)
        return(batch_tensor)

    def compute_output_shape(self, input_shape):

        if(input_shape[0] != None):
            batch_size = int(input_shape[0]/self.pool_ratio[0])
        else:
            batch_size = input_shape[0]
        width = int(input_shape[1]/self.pool_ratio[1])
        height = int(input_shape[2]/self.pool_ratio[2])
        channels = int(input_shape[3]/self.pool_ratio[3])
        return(batch_size, width, height, channels)

    def get_config(self):
        config = {'pooling_ratio': self.pool_ratio, 'pseudo_random': self.pseudo_random,
                  'overlap': self.overlap, 'name': self.name}
        base_config = super(FractionalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
