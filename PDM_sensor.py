import os
import sys, time
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import serial, serial.tools.list_ports
import pyqtgraph
from pyqtgraph import plot
from sklearn import metrics
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QDir, QThread
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pathlib

# Find COM port
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


class Ui_MainWindow (QtWidgets.QMainWindow):
    
    def __init__ (self):
        super(Ui_MainWindow, self).__init__()


        uic.loadUi('window.ui', self)

        self.start = False
        self.save = False

        self.pushButton_openFile.clicked.connect(self.openFile)
        self.pushButton_path.clicked.connect(self.get_path)
        self.pushButton_createModel.clicked.connect(self.create_model)
        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_save.clicked.connect(self.save_data)        
        self.pushButton_play.clicked.connect(self.start_record)
        self.pushButton_stop.clicked.connect(self.collect_stop)
        self.pushButton_connect.clicked.connect(self.connect)

        self.activation = ['sigmoid', 'relu', 'softmax', 'tanh']
        self.optimizer = ['Adam', 'Adagrad', 'Adamax', 'Nadam', 'Optimizer', 'RMSprop', 'SGD']
        self.loss = ['mse', 'mae']
        self.metrics = ['loss', 'accuracy']     
        self.baudrate.addItems(['110', '300', '600', '1200', '2400', '4800', '9600', '14400', '19200', '28800', 
        '31250', '38400', '51200', '56000', '57600', '76800', '115200', '128000', '230400', '256000', '921600'])
        self.baudrate.setCurrentText('9600')
        self.Activation.addItems(self.activation)
        self.Optimizer.addItems(self.optimizer)
        self.Loss.addItems(self.loss)
        self.Metrics.addItems(self.metrics)
        
        self.portlist=find_USB_device()
        self.items=[p[0] for p in self.portlist]#["COM1","COM2"]
        self.serial=None
        self.typeBox_port.addItems(self.items)#database getMotionType()
        self.typeBox_port.setCurrentIndex(self.typeBox_port.count()-1)
        
        self.port.addItems(self.items)

    def start_record(self):
        self.start = True
        # Check numbers of samples
        n_samples = 0
        self.sample = int(self.samples.text())

        if(self.start == True):
            self.baud = self.baudrate.currentText()
            try:    
                self.ser = serial.Serial(self.port.currentText(), self.baud)
            except:
                print('Cannot open port')
            print('Connected')

        if self.save == False:
            while self.ser.isOpen() and n_samples < self.sample: 
                #receive data
                data = self.ser.readline().decode('utf-8').strip()
                data = data[2:][:-5].split(",")
                data_arr = np.asarray([float(_) for _ in data])
                n_samples += 1

                self.plot(data_arr)
        else:
            pass

    def collect_stop(self):
        self.start = False
        self.ser.close()
        print('Disconnected')

    def plot(self, data):
        self.graphWidget.plot(data, clear = True)
        pyqtgraph.QtWidgets.QApplication.processEvents()
    
    # Get directory to save data
    def get_path(self):
        self.dirpath = QDir.currentPath()
        self.file_path = QFileDialog.getExistingDirectory(self, 'Choose directory', self.dirpath)
        self.path_2.setText(self.file_path)

    # Plot saved data
    def openFile(self):
        # Get saved file directory
        fileName, _ = QFileDialog.getOpenFileName(self,"Open File", "","All Files (*);;csv Files (*.csv)")        
        if fileName:
            self.path.setText(fileName)
        # Plot saved data
        data = pd.read_csv(fileName)
        data.plot()
        plt.savefig('plotter.png')
        plotter = QPixmap('plotter.png')
        self.label_14.setPixmap(plotter)        
    
    def save_data(self):
        self.save = True
        # Create the save path
        fileName = str(self.filename.text())
        if not fileName.endswith('.txt'):
            fileName += '.txt'
        save_path = os.path.join(self.file_path, fileName)

        if self.save == True:
            line = 0
            while line < self.sample:
                data = self.ser.readline().decode('utf-8').strip()
                data = data[2:][:-5].split(",")
                data_arr = np.asarray([float(_) for _ in data])

                with open(save_path, 'ab') as file:
                    np.savetxt(file, data_arr)

                self.plot(data_arr)
                line += 1

            print("Saved")

    def create_model(self):
        # build the functional model and train it
        # layers = [50,15,119]
        shape = ((714))
        size = self.size.text()
        layers = size.split(',')
        
        inputs = tf.keras.Input(shape=shape)

        x = inputs
        for layer_size in layers:
            x = tf.keras.layers.Dense(layer_size, activation = 'relu')(x)

        model = tf.keras.Model(inputs = inputs, outputs = x)
        model.compile(optimizer = 'RMSprop', loss = 'mse', metrics = ['accuracy'])

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.label_6.setText(short_model_summary)
  
        self.Model = model           

    def train(self):
        SEED = 1337
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        GESTURES = [
                "punch",
                "flex",    
        ]
        SAMPLES_PER_GESTURE = 119
        NUM_GESTURES = len(GESTURES)
        ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)
        inputs = []
        outputs = []
        for gesture_index in range(NUM_GESTURES):
            gesture = GESTURES[gesture_index]         
            output = ONE_HOT_ENCODED_GESTURES[gesture_index]            
            df = pd.read_csv("C:/VSCode/" + gesture + ".csv")
            num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)            
            
            for i in range(num_recordings):
                tensor = []
                for j in range(SAMPLES_PER_GESTURE):
                    index = i * SAMPLES_PER_GESTURE + j
                    tensor += [
                        (df['aX'][index] + 4) / 8,
                        (df['aY'][index] + 4) / 8,
                        (df['aZ'][index] + 4) / 8,
                        (df['gX'][index] + 2000) / 4000,
                        (df['gY'][index] + 2000) / 4000,
                        (df['gZ'][index] + 2000) / 4000
                    ]

                inputs.append(tensor)
                outputs.append(output)

        inputs = np.array(inputs)
        outputs = np.array(outputs)        
        num_inputs = len(inputs)
        randomize = np.arange(num_inputs)
        np.random.shuffle(randomize)
        inputs = inputs[randomize]
        outputs = outputs[randomize]
        TRAIN_SPLIT = int(0.6 * num_inputs)
        TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)
        inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
        outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])
        history = self.Model.fit(inputs_train, outputs_train, epochs=100, batch_size=1, validation_data=(inputs_validate, outputs_validate))

        val_loss = history.history['val_loss']
        epochs = range(1, len(val_loss) + 1)
        self.widget.plot(epochs, val_loss, label = 'validation loss')

        predictions = self.Model.predict(inputs_test)
        print(predictions)
        test = np.argmax(outputs_test,axis=1)
        pred = np.argmax(predictions,axis=1)
        metrics.confusion_matrix(test,pred)
        data = {'y_true': test,
                'y_pred': pred
        }
        df = pd.DataFrame(data, columns=['y_true','y_pred'])
        confusion_matrix = pd.crosstab(df['y_true'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
        cfm_plot = sn.heatmap(confusion_matrix, annot=True)
        cfm_plot.figure.savefig('cfm.png')

        pixmap = QPixmap('cfm.png')
        self.label_7.setPixmap(pixmap)

        # Convert model to tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.Model)
        tflite_model = converter.convert()
        open("gesture_model.tflite", "wb").write(tflite_model)

        # Convert tflite model to header file
        os.system('echo "const unsigned char model[] = {" > model.h')
        os.system('cat gesture_model.tflite | xxd -i      >> model.h')
        os.system('echo "};"                              >> model.h')

    def connect(self):
        
        self.desc.setText("")
        self.desc.setText(">> trying to connect to port %s ..." % self.typeBox_port.currentText())
        #with serial.Serial(self.typeBox.currentText(), 115200, timeout=1) as self.serial:
        if self.serial is None:
            self.serial=serial.Serial(self.typeBox_port.currentText(), 115200, timeout=1)
            time.sleep(0.05)
            #self.serial.write(b'hello')
            answer=self.readData()
            if answer!="":
                self.desc.setText(self.desc.toPlainText()+"\n>> Connected!\n"+answer)
        else:
            self.desc.setText(">> {} already Opened!\n".format(self.typeBox_port.currentText()))

    def readData(self):
        #self.serial.flush() # it is buffering. required to get the data out *now*
        answer = ""
        while self.serial.inWaiting() > 0: #self.serial.readable() and
            
            print(self.serial.inWaiting())
            answer += "\n"+str(self.serial.readline()).replace("\\r","").replace("\\n","").replace("'","").replace("b","")
            print(self.serial.inWaiting())
        self.desc.setText(self.desc.toPlainText()+"\n"+answer)
        return answer        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    app.exec()