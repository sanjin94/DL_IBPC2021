import csv
import pandas as pd
import numpy as np
R = 4 # 1 -- _d123; 2 -- _d137b; 3 -- _dlab; 4 -- _d205_wdw; 5 -- _d210_wdw
TV = 0.75 # training/validation ratio


file = open("data_raw/{}.csv".format(R))
read = csv.reader(file, delimiter=';')
_read = pd.DataFrame(list(read)).to_numpy()
data_start = 1
data_end = len(_read)
q = _read[:, 1][data_start:data_end].astype(np.float) * -1
ti = _read[:, 2][data_start:data_end].astype(np.float)
te = _read[:, 3][data_start:data_end].astype(np.float)
file.close()
q = q.reshape(-1, 1)
ti = ti.reshape(-1, 1)
te = te.reshape(-1, 1)
