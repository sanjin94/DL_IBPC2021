import matplotlib.pyplot as plt
import numpy as np

import data

train_validation = data.TV

plt.plot(data.te)
plt.plot(data.ti)
plt.plot(data.q)
plt.axvline(x=len(data.ti[0:int(np.round(train_validation * len(data.ti)))]), color='m')
plt.show()
