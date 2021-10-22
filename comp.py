import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import data


ti = data.ti
te = data.te
q = data.q
merged_array = np.stack([ti, te], axis=1)

path_25 = './saved_model_room-{}-0.25'.format(data.R)
path_33 = './saved_model_room-{}-0.33'.format(data.R)
path_50 = './saved_model_room-{}-0.5'.format(data.R)
path_66 = './saved_model_room-{}-0.66'.format(data.R)
path_75 = './saved_model_room-{}-0.75'.format(data.R)

model_25 = load_model(path_25, compile=True)
model_33 = load_model(path_33, compile=True)
model_50 = load_model(path_50, compile=True)
model_66 = load_model(path_66, compile=True)
model_75 = load_model(path_75, compile=True)

#Predictions
predictions_25 = model_25.predict(merged_array)#[:, 1]
predictions_33 = model_33.predict(merged_array)#[:, 1]
predictions_50 = model_50.predict(merged_array)#[:, 1]
predictions_66 = model_66.predict(merged_array)#[:, 1]
predictions_75 = model_75.predict(merged_array)#[:, 1]

# Generating plots (ground truth)
plt.rcParams["figure.figsize"] = (13, 8)
#plt.rcParams["legend.loc"] = 'best'
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.plot(te, label="Exterior temperature $[°C]$")
plt.plot(ti, label="Interior temperature $[°C]$")
plt.plot(q, label="HFM heat flux $[W/m^2]$")

# Generating plots (predictions)
plt.plot(predictions_25, label="tv = 1/4")
plt.plot(predictions_33, label="tv = 1/3")
plt.plot(predictions_50, label="tv = 1/2")
plt.plot(predictions_66, label="tv = 2/3")
plt.plot(predictions_75, label="tv = 3/4")

# Generating plots (additional)
plt.ylabel("Heat flux $[W/m^2]$ / Temperature $[°C]$", fontsize=23)
plt.xlabel("Measuring samples (sampling = every 10 minutes)", fontsize=23)
#plt.legend(fontsize=17, ncol=2)
plt.savefig('room-{}.png'.format(data.R), format='png', dpi=600)
plt.show()
