import numpy as np


def create_dataset(df, time_offset):
    x = []
    y = []
    for i in range(time_offset, df.shape[0]):
        x.append(df[i-time_offset:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y
