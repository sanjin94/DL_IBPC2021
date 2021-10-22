import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import regularizers

import data


# Generating the data
train_validation = data.TV

ti_train = data.ti[0:int(np.round(train_validation*len(data.ti)))]
te_train = data.te[0:int(np.round(train_validation*len(data.ti)))]
q_train = data.q[0:int(np.round(train_validation*len(data.ti)))]

# Making the model
tf.keras.backend.set_floatx('float64')
merged_array = np.stack([ti_train, te_train], axis=1)

input_shape = merged_array.shape
target_shape = q_train.shape


model = Sequential()

model.add(LSTM(units=128, kernel_regularizer=regularizers.L1L2(1e-5)))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

model.fit(merged_array, q_train, epochs=1000)

model.summary()
model.get_config()

filepath = './saved_model_room-{}-{}'.format(data.R, train_validation)
save_model(model, filepath)
