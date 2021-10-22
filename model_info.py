import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import data


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_error(actual, predicted)))


def metrics(actual: np.ndarray, predicted: np.ndarray):
    print("RMSE: ", rmse(actual, predicted))
    print("MSE: ", mse(actual, predicted))
    print("MAE: ", mae(actual, predicted))

train_validation = data.TV

ti_train = data.ti[0:int(np.round(train_validation*len(data.ti)))]
te_train = data.te[0:int(np.round(train_validation*len(data.ti)))]
q_train = data.q[0:int(np.round(train_validation*len(data.ti)))]


ti_validation = data.ti[int(np.round(train_validation * len(data.ti))):len(data.ti)]
te_validation = data.te[int(np.round(train_validation * len(data.ti))):len(data.ti)]
q_validation = data.q[int(np.round(train_validation * len(data.ti))):len(data.ti)]

merged_array = np.stack([ti_train, te_train], axis=1)
merged_array1 = np.stack([ti_validation, te_validation], axis=1)


filepath = './saved_model_room-{}-{}'.format(data.R, train_validation)
model = load_model(filepath, compile=True)

predictions = model.predict(merged_array)#[:, 1]
predictions1 = model.predict(merged_array1)#[:, 1]
result = np.append(predictions, predictions1)

x = np.append(q_train, q_validation)

print("Whole sequence")
metrics(x, result)

plt.rcParams["figure.figsize"] = (13, 8)
plt.rcParams["legend.loc"] = 'upper left'
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
x_train = q_train.flatten()
y_train = predictions.flatten()
x_val = q_validation.flatten()
y_val = predictions1.flatten()
plt.scatter(x_train, y_train, c='r', label="Training")
plt.scatter(x_val, y_val, c='b', label="Validation")
plt.plot(x,x, label="y = x")
plt.ylabel("Predicted", fontsize=14)
plt.xlabel("Actual", fontsize=14)
plt.legend()
plt.savefig('room-{}-{}.png'.format(data.R, train_validation), format='png', dpi=600)
plt.show()
