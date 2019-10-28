# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
from __future__ import absolute_import
import random
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf

class FractionalPooling2D(Layer):
	def __init__(self, pool_ratio = None, pseudo_random = False, overlap = False, name ='FractionPooling2D', **kwargs):
		self.pool_ratio = pool_ratio
		self.input_spec = [InputSpec(ndim=4)]
		self.pseudo_random = pseudo_random
		self.overlap = overlap
# 		self.name = name
		super(FractionalPooling2D, self).__init__(**kwargs)
		
	def call(self, input):
		[batch_tensor,row_pooling,col_pooling] = tf.nn.fractional_max_pool(input, pooling_ratio = self.pool_ratio, pseudo_random = self.pseudo_random, overlapping = self.overlap, seed = 1)
		return(batch_tensor)
		
	def compute_output_shape(self, input_shape):
	
# 		if(K.image_dim_ordering() == 'channels_last' or K.image_dim_ordering() == 'tf'):
			if(input_shape[0] != None):
				batch_size = int(input_shape[0]/self.pool_ratio[0])
			else:
				batch_size = input_shape[0]
			width = int(input_shape[1]/self.pool_ratio[1])
			height = int(input_shape[2]/self.pool_ratio[2])
			channels = int(input_shape[3]/self.pool_ratio[3])
			return(batch_size, width, height, channels)
		
	def get_config(self):
		config = {'pooling_ratio': self.pool_ratio, 'pseudo_random': self.pseudo_random, 'overlap': self.overlap}
		base_config = super(FractionalPooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
		
	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]


#%%
import numpy
import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,Conv1D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[0:49984]
y_train = y_train[0:49984]
X_test = X_test[0:9984]
y_test = y_test[0:9984]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), batch_input_shape=(64, 32, 32, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 4
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
# Block 5
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 6
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
model.add(Flatten())
# fc layer_1
model.add(Dense(4096))
model.add(LeakyReLU(alpha = 0.3))
# fc_layer_2
model.add(Dense(4096))
model.add(LeakyReLU(alpha = 0.3))

model.add(Dense(num_classes, activation='softmax'))

# opt = keras.optimizers.Adadelta(0.1,decay=1e-4)

model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint('Model.hdf5', monitor='val_loss', save_best_only = True, verbose=1, mode='min')

callbacks_list = [checkpoint]
#model.load_weights('Model.hdf5')
epochs = 1000
model.fit(X_train, y_train, validation_data = [X_test,y_test], nb_epoch=epochs, batch_size=64, callbacks=callbacks_list)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#%%



