import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt

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

num_res_net_blocks = 10  # number of resnet blocks

# Function to create a Residual block
def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


## Creating the model

# 1. Input block
inputs = keras.Input(shape=(24, 24, 3)) 
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x) # First convolutional layer
x = layers.MaxPooling2D((3, 3), padding = 'SAME')(x) # Pooling Layer

# 2. Add residual blocks
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

# 3. Last stage of convolutional layer
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Flatten()(x)

# 4. Dense Blocks after Convolutional block
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Dropout layer 
outputs = layers.Dense(10, activation='softmax')(x) # outputs of the model

# create the model
res_net_model = keras.Model(inputs, outputs)


# Callbacks to take care of during training
callbacks = [
    # Write TensorBoard logs to `./logs` directory
    keras.callbacks.TensorBoard(
        log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
]

# compile the model
res_net_model.compile(optimizer=keras.optimizers.Adam(),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc'])

# fit the model on the training dataset
epochs = 10
res_net_model.fit(X_train, y_train, validation_data = [X_test,y_test], nb_epoch=epochs, batch_size=64, callbacks=callbacks_list)
