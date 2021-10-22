import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras import regularizers

import data


# Generating the data
train_validation = data.TV

ti_train = data.ti[12:int(np.round(train_validation*len(data.ti)))]
te_train = data.te[12:int(np.round(train_validation*len(data.ti)))]
q_train = data.q[12:int(np.round(train_validation*len(data.ti)))]

# Making the model
tf.keras.backend.set_floatx('float64')
merged_array = np.stack([ti_train, te_train], axis=1)

model = Sequential()

model.add(LSTM(units=16, kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001)))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(merged_array, q_train, epochs=7000)

model.summary()
model.get_config()

filepath = './saved_model_room-{}-{}'.format(data.R, train_validation)
save_model(model, filepath)
