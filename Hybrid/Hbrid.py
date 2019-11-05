#importing required libraries
from __future__ import absolute_import
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#defining FMP class
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

#loading dataset
#-------------------------------------------------------------------------#
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(64).shuffle(50000)
train_dataset = train_dataset.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.repeat()
valid_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.repeat()

#defining blocks
def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size,
                      activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

# making the model
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = FractionalPooling2D(3)(x)

num_res_net_blocks = 10

for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)
    
x = layers.Conv2D(64, 3, activation='relu')(x)
x = FractionalPooling2D(3)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
hybrid_model = keras.Model(inputs, outputs)

hybrid_model.compile(optimizer=keras.optimizers.Adam(),
                     loss='categorical_crossentropy',
                     metrics=['acc'])

print(hybrid_model.summary())

checkpoint = ModelCheckpoint(
    'Model.hdf5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')

callbacks_list = [checkpoint]
results = hybrid_model.fit(train_dataset, epochs=50, steps_per_epoch=100,
                           validation_data=valid_dataset,
                           validation_steps=3, callbacks=callbacks_list)


#---------------------------------------------------------------#
#plotting
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#----------------------------------------------------------------#
%matplotlib inline
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
