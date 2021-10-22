import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import data


# Generating the data
train_validation = 1/2

ti_train = data.ti[0:int(np.round(train_validation*len(data.ti)))]
te_train = data.te[0:int(np.round(train_validation*len(data.ti)))]
q_train = data.q[0:int(np.round(train_validation*len(data.ti)))]

ti_validation = data.ti[int(np.round(train_validation * len(data.ti))):len(data.ti)]
te_validation = data.te[int(np.round(train_validation * len(data.ti))):len(data.ti)]
q_validation = data.q[int(np.round(train_validation * len(data.ti))):len(data.ti)]

merged_array = np.stack([ti_train, te_train], axis=1)
merged_array1 = np.stack([ti_validation, te_validation], axis=1)

filepath = './saved_model_room-{}'.format(data.R)
model = load_model(filepath, compile=True)

predictions = model.predict(merged_array)[:, 1]
predictions1 = model.predict(merged_array1)[:, 1]

plt.plot(data.te, label="Exterior temperature $[°C]$")
plt.plot(data.ti, label="Interior temperature $[°C]$")
plt.plot(data.q, label="HFM heat flux $[W/m^2]$")
result = np.append(predictions, predictions1)
plt.plot(result, label="MLP heat flux $[W/m^2]$")
plt.axvline(x=len(data.ti[0:int(np.round(train_validation * len(data.ti)))]), color='m')

plt.show()
