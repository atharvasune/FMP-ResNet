from __future__ import absolute_import
import matplotlib
from tensorflow.keras.layers import Conv2D, Conv1D, LSTM
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
import tensorflow.keras as keras
import numpy
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


#--------------------------------------------------------------#


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
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


# Function to create a normal convolutional block // Removed from RESNET and added here
def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size,
                      activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x


# Create the model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), batch_input_shape=(64, 32, 32, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.6, 1.6, 1), pseudo_random=True, overlap=True))
# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.25, 1.25, 1), pseudo_random=True, overlap=True))
# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.6, 1.6, 1), pseudo_random=True, overlap=True))
# Block 4
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.25, 1.25, 1), pseudo_random=True, overlap=True))
# Block 5
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.6, 1.6, 1), pseudo_random=True, overlap=True))
# Block 6
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(FractionalPooling2D(pool_ratio=(
    1, 1.25, 1.25, 1), pseudo_random=True, overlap=True))
model.add(Flatten())
# fc layer_1
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.3))
# fc_layer_2
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(10, activation='softmax'))

opt = keras.optimizers.Adadelta(0.1, decay=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint(
    'Model.hdf5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')

callbacks_list = [checkpoint]
# model.load_weights('Model.hdf5')
results = model.fit(train_dataset, epochs=50, steps_per_epoch=100,
                    validation_data=valid_dataset,
                    validation_steps=3, callbacks=callbacks_list)
#----------------------------------------------------------------------#
%matplotlib inline
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
