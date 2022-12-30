import serial, serial.tools.list_ports
from PyQt5.QtWidgets import *
import numpy as np
import os, pathlib
from PyQt5.QtCore import QDir, QThread, QMutex, QObject, pyqtSignal
import pyqtgraph as pg
from pyqtgraph import plot
import pandas as pd



import IMU_Capture_UI

def find_USB_device(USB_DEV_NAME=None): 
    myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
    print(myports)
    usb_port_list = [p[0] for p in myports]
    usb_device_list = [p[1] for p in myports]
    print(usb_device_list)

    if USB_DEV_NAME is None:
        return myports
    else:
        USB_DEV_NAME=str(USB_DEV_NAME).replace("'","").replace("b","")
        for device in usb_device_list:
            print("{} -> {}".format(USB_DEV_NAME,device))
            print(USB_DEV_NAME in device)
            if USB_DEV_NAME in device:
                print(device)
                usb_id = device[device.index("COM"):device.index("COM")+4]
            
                print("{} port is {}".format(USB_DEV_NAME,usb_id))
                return usb_id

class Save_raw_data(QObject):
    finished = pyqtSignal()
    mutex = QMutex

    def __init__(self):
        super().__init__()

    def save_data(self, data, filename):

        _filename = filename

        if not _filename.endswith('.csv'):
            _filename += '.csv'

        file = open(_filename, 'a')

        file.write(data + '\n')

        file.close()


class Record(IMU_Capture_UI.Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.mainWindow = QMainWindow()
        self.setupUi(self.mainWindow)
       
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_getpath.clicked.connect(self.get_path)
        self.pushButton_stop.clicked.connect(self.stop)
        self.pushButton_openFile.clicked.connect(self.open_file)

        self.baudrate.addItems(['9600', '115200'])
        self.baudrate.setCurrentText('115200')
        self.fileName.setText('test.csv')
        self.path.setText(str(os.getcwd()))
        self.Samples.setText('100')
        
        self.baud = self.baudrate.currentText()
        self.portlist = find_USB_device()
        self.items = [p[0] for p in self.portlist]
        self.serial = None
        # self.serial = serial.Serial(self.port.currentText(), self.baud)       
        self.port.addItems(self.items)

    def start(self):

        try: 
            self.ser = serial.Serial(self.port.currentText(), self.baud)
        except:
            print('Cannot open port')
        print('Connected')

        line = 0
        windowwidth = 100
        X = np.linspace(0, 0, windowwidth)

        while self.ser.isOpen() and line <= 49:

            getData = str(self.ser.readline())
            self.data = getData[2:][:-5]
            data = self.data.split(',')
            print(self.data)
            
            data_arr = np.asarray([data])
            
            X[:-1] = X[1:]
            X[-1] = data_arr
            self.graphicsView.clear()
            self.plot(X)
                
            if self.checkBox_saving.isChecked():
                thread = self.create_thread(self.data, filename = self.fileName.text())
                thread.start()

                thread.quit()
                thread.wait()

            line = line + 1

        print('Completed')

    def create_thread(self, data, filename):
        thread = QThread()
        worker = Save_raw_data()
        worker.moveToThread(thread)
        thread.started.connect(lambda: worker.save_data(data, filename))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return thread

    def plot(self, data):
        self.graphicsView.setBackground('w')
        self.graphicsView.showGrid(x = True, y = True)
        self.graphicsView.plot(data, clear = False)
        pg.QtWidgets.QApplication.processEvents()

    def get_path(self):
        self.dirpath = QDir.currentPath()
        self.file_path= QFileDialog.getExistingDirectory(None, 'Choose directory', self.dirpath)
        self.path.setText(self.file_path)

    def open_file(self):
        filename = self.fileName.text()

        if not filename.endswith('.csv'):
            filename += '.csv'
        path = os.path.join(self.path.text(), filename)
        print(path)
        data = pd.read_csv(path)
        data = data.transpose()
        pen = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(data.shape[0]):
            self.plot(data.iloc[i])
    
    def stop(self):
        self.ser.close()
        print('Closed the port!')
        

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Record()
    window.mainWindow.show()
    app.exec()    