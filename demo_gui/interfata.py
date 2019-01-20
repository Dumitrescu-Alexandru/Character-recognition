# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from train import train_network
from saver import Saver
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from get_prediction import Predictions
import scipy.misc
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject, QPoint, QRectF, QRect
from PyQt5.QtGui import QPainterPath, QPainter, QPen, QPixmap, QScreen, QColor, QBrush
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QApplication, QSizePolicy
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import ImageGrab
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl


class Drawer(QtWidgets.QWidget):
    newPoint = pyqtSignal(QPoint)
    color = QColor("black")
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.path = QPainterPath()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(7)
        pen.setColor(self.color)
        painter.setPen(pen)
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        self.newPoint.emit(event.pos())
        self.update()

    def draw(self):
        self.color = QColor("black")
        self.path = QPainterPath()

    def erase(self):
        self.color = QColor("white")
        self.path = QPainterPath()

class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum,QtWidgets.QSizePolicy.Maximum))
        Canvas.updateGeometry(self)
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.loaded_images = None
        self.loaded_lbls = None
        MainWindow.setObjectName("OCR Engine Tester")
        self.modified_image_predict = None
        self.image_to_predict = None
        MainWindow.resize(818, 737)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 821, 671))
        self.tabWidget.setStyleSheet("background-color:#ffe0b3;")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 741, 51))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 137, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 10, 137, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(370, 10, 121, 21))
        self.label.setObjectName("label")
        self.textEdit_3 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_3.setGeometry(QtCore.QRect(500, 10, 41, 31))
        self.textEdit_3.setStyleSheet("background-color:white;")
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.label.raise_()
        self.textEdit_3.raise_()
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 60, 741, 291))
        self.groupBox_2.setObjectName("groupBox_2")

        self.widget = QtWidgets.QWidget(self.groupBox_2)
        self.drawer = Drawer(self.widget)

        self.widget.setGeometry(QtCore.QRect(10, 30, 250, 250))
        self.widget.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"border: 2px solid;")
        self.widget.setObjectName("widget")
        self.widget.setMouseTracking(True)
        self.widget.setAutoFillBackground(False)
        self.widget.setStyleSheet("background-color:rgb(255, 255, 255); border: 2px solid;")
        self.widget.setObjectName("widget")
        self.widget.setLayout(QVBoxLayout())
        self.widget.layout().addWidget(self.drawer)



        self.widget_2 = QtWidgets.QWidget(self.groupBox_2)
        self.widget_2.setGeometry(QtCore.QRect(310, 30, 250, 250))
        self.widget_2.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"border: 2px solid;")
        self.widget_2.setObjectName("widget_2")
        self.widget_2.canvas = MplCanvas()
        self.widget_2.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.widget_2.vbl.addWidget(self.widget_2.canvas)
        self.widget_2.setLayout(self.widget_2.vbl)
        self.widget_2.canvas.ax.axis('off')
        #self.widget_2.canvas.ax.set_xlim(0, 40)

        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_4.setGeometry(QtCore.QRect(590, 30, 121, 181))
        self.groupBox_4.setObjectName("groupBox_4")
        self.poison_noise = QtWidgets.QCheckBox(self.groupBox_4)
        self.poison_noise.setGeometry(QtCore.QRect(30, 40, 81, 17))
        self.poison_noise.setObjectName("poison_noise")
        self.speckle_noise = QtWidgets.QCheckBox(self.groupBox_4)
        self.speckle_noise.setGeometry(QtCore.QRect(30, 100, 91, 17))
        self.speckle_noise.setObjectName("speckle_noise")
        '''
        self.progress_bar = QtWidgets.QSlider(self.groupBox_4)
        self.progress_bar.setGeometry(QtCore.QRect(30, 140, 71, 20))
        self.progress_bar.setOrientation(QtCore.Qt.Horizontal)
        self.progress_bar.setObjectName("progress_bar")
        '''
        self.gaussian_noise = QtWidgets.QCheckBox(self.groupBox_4)
        self.gaussian_noise.setGeometry(QtCore.QRect(30, 70, 81, 17))
        self.gaussian_noise.setObjectName("gaussian_noise")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 350, 741, 311))
        self.groupBox_3.setObjectName("groupBox_3")
        self.widget_3 = QtWidgets.QWidget(self.groupBox_3)
        self.widget_3.setGeometry(QtCore.QRect(10, 30, 551, 250))
        self.widget_3.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"border: 2px solid;")
        self.widget_3.setObjectName("widget_3")
        self.widget_3.canvas = MplCanvas()
        self.widget_3.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.widget_3.vbl.addWidget(self.widget_3.canvas)
        self.widget_3.setLayout(self.widget_3.vbl)
        self.widget_3.canvas.ax.axis('off')

        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_6.setGeometry(QtCore.QRect(590, 170, 137, 23))
        self.pushButton_6.setObjectName("pushButton_6")


        self.original = QtWidgets.QRadioButton(self.groupBox_3)
        self.original.setGeometry(QtCore.QRect(620, 80, 82, 21))
        self.original.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.original.setObjectName("original")
        self.trained = QtWidgets.QRadioButton(self.groupBox_3)
        self.trained.setGeometry(QtCore.QRect(620, 120, 82, 17))
        self.trained.setObjectName("trained")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setEnabled(True)
        self.tab_2.setObjectName("tab_2")
        self.listWidget = QtWidgets.QListWidget(self.tab_2)
        self.listWidget.setGeometry(QtCore.QRect(60, 80, 137, 250))
        self.listWidget.setStyleSheet("background-color:white;\n"
"")
        self.listWidget.setObjectName("listWidget")
        self.widget_4 = QtWidgets.QWidget(self.tab_2)
        self.widget_4.setGeometry(QtCore.QRect(430, 80, 250, 250))
        self.widget_4.canvas = MplCanvas()
        self.widget_4.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.widget_4.vbl.addWidget(self.widget_4.canvas)
        self.widget_4.setLayout(self.widget_4.vbl)
        self.widget_4.canvas.ax.axis('off')
        self.widget_4.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"border: 2px solid;")
        self.widget_4.setObjectName("widget_4")
        self.show_images = QtWidgets.QPushButton(self.tab_2)
        self.show_images.setGeometry(QtCore.QRect(490, 20, 137, 51))
        self.show_images.clicked.connect(self.show_image)
        self.show_images.setStyleSheet("#show_images {\n"
"font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;\n"
"font-size:15px;\n"
"color:#1a6600;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"background-color:#ffad33;\n"
"}\n"
"#show_images:pressed {\n"
"    border: 2px solid#black;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"    background-color:#b30000;\n"
"    color:white;\n"
"}\n"
"#show_images:hover{\n"
"        background-color:#cc7a00;\n"
"    border-style: solid;\n"
"    border-width: 2px ;\n"
"    border-color:#1a6600;\n"
"        \n"
"}")
        self.show_images.setObjectName("show_images")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(550, 340, 47, 13))
        self.label_2.setObjectName("label_2")
        self.list_images = QtWidgets.QPushButton(self.tab_2)
        self.list_images.setGeometry(QtCore.QRect(60, 20, 137, 51))
        self.list_images.setStyleSheet("#list_images {\n"
"font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;\n"
"font-size:15px;\n"
"color:#1a6600;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"background-color:#ffad33;\n"
"}\n"
"#list_images:pressed {\n"
"    border: 2px solid#black;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"    background-color:#b30000;\n"
"    color:white;\n"
"}\n"
"#list_images:hover{\n"
"        background-color:#cc7a00;\n"
"    border-style: solid;\n"
"    border-width: 2px ;\n"
"    border-color:#1a6600;\n"
"        \n"
"}")
        self.list_images.setObjectName("list_images")
        self.delete_images_btn = QtWidgets.QPushButton(self.tab_2)
        self.delete_images_btn.setGeometry(QtCore.QRect(60, 350, 137, 51))
        self.delete_images_btn.setObjectName("delete_images_btn")
        self.delete_images_btn.setStyleSheet("#delete_images_btn {\n"
 "font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;\n"
 "font-size:15px;\n"
 "color:#1a6600;\n"
 "    padding: 10px;\n"
 "    border-radius: 20px 10px;\n"
 "background-color:#ffad33;\n"
 "}\n"
 "#delete_images_btn:pressed {\n"
 "    border: 2px solid#black;\n"
 "    padding: 10px;\n"
 "    border-radius: 20px 10px;\n"
 "    background-color:#b30000;\n"
 "    color:white;\n"
 "}\n"
 "#delete_images_btn:hover{\n"
 "        background-color:#cc7a00;\n"
 "    border-style: solid;\n"
 "    border-width: 2px ;\n"
 "    border-color:#1a6600;\n"
 "        \n"
 "}")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_5.setGeometry(QtCore.QRect(70, 430, 621, 131))
        self.groupBox_5.setObjectName("groupBox_5")
        self.refresh = QtWidgets.QPushButton(self.groupBox_5)
        self.refresh.setGeometry(QtCore.QRect(460, 40, 137, 51))
        self.refresh.clicked.connect(self.refresh_fcn)
        self.refresh.setStyleSheet("#refresh {\n"
"font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;\n"
"font-size:15px;\n"
"color:#1a6600;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"background-color:#ffad33;\n"
"}\n"
"#refresh:pressed {\n"
"    border: 2px solid#black;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"    background-color:#b30000;\n"
"    color:white;\n"
"}\n"
"#refresh:hover{\n"
"        background-color:#cc7a00;\n"
"    border-style: solid;\n"
"    border-width: 2px ;\n"
"    border-color:#1a6600;\n"
"        \n"
"}\n"
"")
        self.refresh.setObjectName("refresh")
        self.textEdit_2 = QtWidgets.QTextEdit(self.groupBox_5)
        self.textEdit_2.setGeometry(QtCore.QRect(110, 70, 81, 31))
        self.textEdit_2.setStyleSheet("background-color:white;")
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_5)
        self.textEdit.setGeometry(QtCore.QRect(110, 30, 81, 31))
        self.textEdit.setStyleSheet("background-color:white;")
        self.textEdit.setObjectName("textEdit")
        self.train = QtWidgets.QPushButton(self.groupBox_5)
        self.train.setGeometry(QtCore.QRect(240, 40, 137, 51))
        self.train.clicked.connect(self.train_fcn)
        self.train.setStyleSheet("#train {\n"
"font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;\n"
"font-size:15px;\n"
"color:#1a6600;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"background-color:#ffad33;\n"
"}\n"
"#train:pressed {\n"
"    border: 2px solid#black;\n"
"    padding: 10px;\n"
"    border-radius: 20px 10px;\n"
"    background-color:#b30000;\n"
"    color:white;\n"
"}\n"
"#train:hover{\n"
"        background-color:#cc7a00;\n"
"    border-style: solid;\n"
"    border-width: 2px ;\n"
"    border-color:#1a6600;\n"
"        \n"
"}")
        self.train.setObjectName("train")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 91, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label_4.setObjectName("label_4")
        self.widget_4.raise_()
        self.show_images.raise_()
        self.label_2.raise_()
        self.list_images.raise_()
        self.delete_images_btn.raise_()
        self.listWidget.raise_()
        self.groupBox_5.raise_()
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 818, 21))
        self.menubar.setObjectName("menubar")
        self.menuNew = QtWidgets.QMenu(self.menubar)
        self.menuNew.setObjectName("menuNew")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.actionDraw = QtWidgets.QAction(MainWindow)
        self.actionDraw.triggered.connect(self.draw)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/img/drawer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDraw.setIcon(icon)
        self.actionDraw.setObjectName("actionDraw")

        self.actionErase = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/img/eraser.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionErase.setIcon(icon1)
        self.actionErase.setObjectName("actionErase")

        self.actionSave_for_training = QtWidgets.QAction(MainWindow)
        self.actionSave_for_training.triggered.connect(self.save_future_training)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/img/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave_for_training.setIcon(icon2)
        self.actionSave_for_training.setObjectName("actionSave_for_training")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)

        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/img/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExit.setIcon(icon3)
        self.actionExit.setObjectName("actionExit")
        self.menuNew.addAction(self.actionDraw)
        self.menuNew.addAction(self.actionErase)
        self.menuNew.addAction(self.actionSave_for_training)
        self.menuNew.addSeparator()
        self.menuNew.addAction(self.actionExit)
        self.menubar.addAction(self.menuNew.menuAction())
        self.toolBar.addAction(self.actionDraw)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionErase)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSave_for_training)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionExit)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "OCR Engine Tester"))
        self.pushButton.setText(_translate("MainWindow", "Save Image"))
        self.pushButton.clicked.connect(self.save_img)

        self.pushButton_2.setText(_translate("MainWindow", "Show Plotted"))
        self.pushButton_2.clicked.connect(self.plot_the_img)

        self.label.setText(_translate("MainWindow", "Label For future training"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Drawing and plotting"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Modify Image"))
        self.poison_noise.setText(_translate("MainWindow", "Poison"))
        self.speckle_noise.setText(_translate("MainWindow", "Speckle"))
        self.gaussian_noise.setText(_translate("MainWindow", "Gauss"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Get Prediction"))

        self.pushButton_6.setText(_translate("MainWindow", "Get Prediction"))
        self.pushButton_6.clicked.connect(self.get_prediction)


        self.original.setText(_translate("MainWindow", "original"))
        self.trained.setText(_translate("MainWindow", "trained"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Predictions"))
        self.show_images.setText(_translate("MainWindow", "Show image"))
        self.label_2.setText(_translate("MainWindow", "-1"))
        self.delete_images_btn.setText(_translate("MainWindow", "Delete Images"))
        self.list_images.setText(_translate("MainWindow", "List images"))
        self.list_images.clicked.connect(self.list_images_fcn)
        self.delete_images_btn.clicked.connect(self.delete_images_fcn)
        self.groupBox_5.setTitle(_translate("MainWindow", "Modify network"))
        self.refresh.setText(_translate("MainWindow", "Refresh"))
        self.train.setText(_translate("MainWindow", "Train"))
        self.label_3.setText(_translate("MainWindow", "No. of steps"))
        self.label_4.setText(_translate("MainWindow", "Train coefficient"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Training"))
        self.menuNew.setTitle(_translate("MainWindow", "File"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionDraw.setText(_translate("MainWindow", "Draw"))
        self.actionErase.setText(_translate("MainWindow", "Erase"))
        self.actionErase.triggered.connect(self.erase)
        self.actionSave_for_training.setText(_translate("MainWindow", "Save for training"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def draw(self):
        self.drawer.draw()
    def erase(self):
        self.drawer.erase()

    def save_img(self):
        filename="poza.jpg"
        screenshot = QtWidgets.QWidget.grab(self.widget)
        screenshot.save(filename, 'jpg')

    def refresh_fcn(self):
        from refresh_networks import refresh
        refresh()
        self.statusbar.showMessage("Successfully refreshed the network.")

    def plot_the_img(self):
        img = cv2.imread('poza.jpg', 0)
        new_image = np.ones(np.shape(img))-img
        new_image = scipy.misc.imresize(new_image, (28, 28))
        self.image_to_predict = new_image
        self.modified_image_predict = new_image

        if self.poison_noise.isChecked():
            new_image = self.poison_noise_fcn(new_image)
        if self.gaussian_noise.isChecked():
            new_image = self.gaussian_noise_fcn(new_image)
        if self.speckle_noise.isChecked():
            new_image = self.speckle_noise_fcn(new_image)

        self.modified_image_predict = new_image
        self.widget_2.canvas.ax.imshow(new_image,cmap=mpl.cm.gray)
        self.widget_2.canvas.ax.axis('off')
        self.widget_2.canvas.draw()

    def train_fcn(self):
        no_steps = self.textEdit.toPlainText()
        train_coefficient = self.textEdit_2.toPlainText()
        aux = False
        for a in no_steps:
            if ord(a)<48 or ord(a)>57:
                aux = True

        if ord(train_coefficient[0])<48 or ord(train_coefficient[0])>57:
            aux = True
        if len(train_coefficient)>2:
            if ord(train_coefficient[1]) != 46:
                aux = True


        for i in range(2,len(train_coefficient)):
            if ord(train_coefficient[i]) < 48 or ord(train_coefficient[i]) > 57:
                aux = True
        if aux == False:
            no_steps = int(no_steps)
            if len(train_coefficient) == 1:
                train_coefficient = int(train_coefficient)
            else:
                tr_coef = int(train_coefficient[0])
                if len(train_coefficient) > 2:
                    for i in range(2,len(train_coefficient)):
                        tr_coef = tr_coef+float(train_coefficient[i])/(10**(i-1))
                train_coefficient = tr_coef
        if aux == True:
            self.statusbar.showMessage("You must enter an integer for no of steps and a float for train coefficient")
        else:
            self.statusbar.showMessage("The network training...")
            train_network(self.loaded_images,self.loaded_lbls,train_coefficient,no_steps)
            self.statusbar.showMessage("The network has successfully trained")


    def delete_images_fcn(self):

        if self.loaded_images == None:
            self.statusbar.showMessage("Please list the images and select one first")
        else:
            index = int(self.listWidget.currentRow())
            import os
            import pickle
            os.remove('imagini_salvate/future_train_imgs.bin')
            os.remove('imagini_salvate/future_train_lbls.bin')
            new_arr_imgs = np.delete(self.loaded_images,index,axis=0)
            new_arr_lbls = np.delete(self.loaded_lbls,index,axis=0)
            img_file_name = 'imagini_salvate/future_train_imgs.bin'
            lbl_file_name = 'imagini_salvate/future_train_lbls.bin'

            file = open(img_file_name, 'ab')
            for img in new_arr_imgs:
                pickle.dump(img, file)
            file.close()

            file = open(lbl_file_name, 'ab')
            for lbl in new_arr_lbls:
                pickle.dump(lbl,file)
            file.close()
            self.list_images_fcn()
            self.statusbar.showMessage("The selected image has been deleted")
    def list_images_fcn(self):
        #self.listView
        self.listWidget.clear()
        load_images = Saver()
        imgs = load_images.read_images('imagini_salvate/')
        lbls = load_images.read_labels('imagini_salvate/')
        for i in range(np.shape(imgs)[0]):
            lbl = lbls[i]
            self.listWidget.addItem("img nr_"+str(i)+"_label_"+str(lbl))
        self.loaded_images = imgs
        self.loaded_lbls = lbls

    def show_image(self):

        if self.loaded_images == None:
            self.statusbar.showMessage("Please list the images and select one first")
        else:
            index = int(self.listWidget.currentRow())
            new_image = self.loaded_images[index]
            self.widget_4.canvas.ax.imshow(new_image, cmap=mpl.cm.gray)
            self.widget_4.canvas.ax.axis('off')
            self.widget_4.canvas.draw()
            self.label_2.setText(str(self.loaded_lbls[index]))

    def get_prediction(self):

        self.widget_3.canvas.ax.cla()
        if self.original.isChecked():
            self.statusbar.showMessage("Predicting with the original network...")
            pred = Predictions()
            predictions = pred.convo_model_prediction(self.modified_image_predict)
            refitted_predictions = self.refit_precitions(predictions)
            numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            numbers = np.array(numbers)
            refitted_predictions = refitted_predictions+100
            #self.widget_3.canvas.ax.set_xlim([0,9])
            self.widget_3.canvas.ax.bar(numbers,refitted_predictions)
            self.widget_3.canvas.ax.set_xticks(numbers, minor=True)
            self.widget_3.canvas.ax.axis('on')
            self.widget_3.canvas.draw()
            prediected = np.argmax(refitted_predictions)
            self.statusbar.showMessage("Predicted " + str(prediected))
        elif self.trained.isChecked():
            self.statusbar.showMessage("Predicting with the trained network...")
            pred = Predictions()
            predictions = pred.convo_model_prediction_interf(self.modified_image_predict)
            refitted_predictions = self.refit_precitions(predictions)
            numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            refitted_predictions = refitted_predictions+100
            #self.widget_3.canvas.ax.set_xlim([0,9])
            self.widget_3.canvas.ax.bar(numbers,refitted_predictions)
            self.widget_3.canvas.ax.set_xticks(numbers, minor=True)
            self.widget_3.canvas.ax.axis('on')
            self.widget_3.canvas.draw()
            prediected = np.argmax(refitted_predictions)
            self.statusbar.showMessage("Predicted " + str(prediected))
        else:
            self.statusbar.showMessage("You need to specify the network.")

        #self.widget_3.canvas.ax.imshow(self.image_to_predict, cmap=mpl.cm.gray)
        #self.widget_3.canvas.ax.axis('off')
        #self.widget_3.canvas.draw()
    def refit_precitions(self,predictions):
        predictions=predictions[0]
        predictions = np.array(predictions)
        min = np.argmin(predictions)
        min = predictions[min]
        if min < 0:
            predictions = predictions - min
        return predictions

    def save_future_training(self):
        aux = False
        a = self.textEdit_3.toPlainText()
        if (a == ""):
            self.statusbar.showMessage("Please enter a label between 0 and 9")
        else:
            if (len(a)>1):
                self.statusbar.showMessage("Please enter a label between 0 and 9")
            else:
                b = ord(a)
                if(b < 48 or b > 57):
                    self.statusbar.showMessage("Please enter a label between 0 and 9")
                else:
                    aux = True
        if self.modified_image_predict is None:
            self.statusbar.showMessage("Please enter a valid image through the Show plotted button")
        elif self.modified_image_predict is not None and aux == True:
            a = int(a)
            saver_img = Saver(self.modified_image_predict,a)
            saver_img.save('imagini_salvate/')
            imgs = saver_img.read_images('imagini_salvate/')
            lbls = saver_img.read_labels('imagini_salvate/')
            self.statusbar.showMessage("Image has been saved for training")
            self.textEdit_3.setText("")

    def gaussian_noise_fcn(self,image,intensity=50):
        image = np.reshape(image, newshape=(28, 28, 1))
        row, col, ch = image.shape
        mean = 0
        var = intensity
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.reshape(noisy,newshape=(28,28))

    def speckle_noise_fcn(self,image):
        image = np.reshape(image,newshape=(28,28,1))
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.2
        return np.reshape(noisy,newshape=(28,28))

    def poison_noise_fcn(self,image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy


import resource

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

