#%%
import serial
import numpy as np
import matplotlib.pyplot as plt

samples = 49
port = 'COM4'
baud = 115200
datas = []

ser = serial.Serial(port, baud)

print('Connected')

windowwidth = 100
X = np.linspace(0, 0, windowwidth)


line = 0
while line <= samples:


    getData = str(ser.readline())

    data=getData[2:][:-5]
    data_arr = data.split(',')
    print(line)
    print(data)
    a = data_arr
    # if line >= 1:
    #     data_arr = np.asarray([float(_) for _ in data])
    #     datas.append(data_arr)
    #     arr = np.array(datas)
        # for i in range(9):
        #     plt.plot(arr[:, i])
    file = open('test_pdm.csv', 'a')
    file.write(data + "\n")

    # data_arr = np.asarray([float(_) for _ in data])

    # for i in range(len(data)):
    #     X[:-1] = X[1:]
    #     X[-1] = data_arr[i]
    #     # plt.plot(X)
    #     # print(data_arr.shape)

    line = line + 1
file.close()

print('completed')
print(a)
# print(X)
# plt.plot(X)

# %%
data_arr = np.asarray([float(_) for _ in a])
print(data_arr)
# %%
plt.plot(data_arr)
# %%
import pandas as pd

df = pd.read_csv('test_pdm.csv')

df.shape

# %%
import matplotlib.pyplot as plt

print(df.loc[0].shape)
# %%
