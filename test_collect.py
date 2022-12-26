#%%
import serial
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


samples = 110
port = 'COM4'
baud = 9600
datas = []


ser = serial.Serial(port, baud)

print('Connected')

windowwidth = 100
X = np.zeros((windowwidth, 9))
ptr = -windowwidth

line = 0
while line <= samples:

    getData = str(ser.readline())
    data=getData[2:][:-5].split(',')
    # print(data)
    if line >= 1:
        data_arr = np.asarray([float(_) for _ in data])
        datas.append(data_arr)
        arr = np.array(datas)
        # for i in range(9):
        #     plt.plot(arr[:, i])

        X[:-1] = X[1:]
        X[-1] = data_arr
        plt.plot(X)
        # print(data_arr.shape)


    line = line + 1
    
# arr = np.array(datas)
# print(arr)
# print(arr.shape)
# plt.subplot(121)
# plt.plot(arr[:,0])
# plt.subplot(122)
# plt.plot(arr[:,3])


# %%
import pandas as pd
path = "punch.csv"

data = pd.read_csv(path)
print(data.loc[0])

# %%
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import pandas as pd
import numpy as np


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('w')

        
        windowwidth = 100
        X = np.zeros((windowwidth, data.shape[1]))

        data = pd.read_csv(path)
        print(data.loc[0])

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                _row = data.loc[row]
                X[:-1] = X[1:]
                X[-1] = _row
                # print(X)
                self.graphWidget.plot(X[col], pen = 'r')    
        # plot data: x, y values
        # self.graphWidget.plot(data['aX'], pen = 'r')

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('punch.csv')
print(data.loc[0])

windowwidth = 100
X = np.zeros((windowwidth, data.shape[1]))
print(X.shape)

for i in range(data.shape[0]):
    X[:-1] = X[1:]
    X[-1] = data.loc[i]
    print(X)
    plt.clf()
    plt.plot(X)

plt.plot(X)
plt.show()

# plt.plot()

# %%
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("Plotting Example")
#mw = QtWidgets.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(500,500)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

data = pd.read_csv('punch.csv')

windowWidth = 100                       
Xm = np.zeros((windowWidth, data.shape[1]))        
ptr = -windowWidth

p1 = win.addPlot(title="Basic array plotting")
# p1.plot(x, y)

p1.plot(data[0])

# for i in range(data.shape[0]):  
#     Xm[:-1] = Xm[1:]                      
#     Xm[-1] = data.loc[i]  
#     p1.plot(Xm[0])
    # for index in range(data.shape[1]): 
    #     p1.plot(Xm[index])

if __name__ == '__main__':
    pg.exec()
# %%
