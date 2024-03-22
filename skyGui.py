'''
河北水勘院遥感监测，图形界面接口Demo。
'''
from PyQt5.Qt import  QThread
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QAction, QMenu, QMessageBox
import sys
import cv2
from PIL import Image
import os
import scipy.io
import numpy as np
from Networks.prediction import Predict
sys.path.append(r'./Networks/CSSSAN')
sys.path.append(r'./Networks/STANet')
sys.path.append(r'./Networks/yolov4')

from Networks.CSSSAN import CSSSAN
from Networks.STANet import STANet
from Networks.yolov4 import yolo

######################tools##################
'''
实现cv类型的图片到Qimage类型图片的转化
log:
    1:
        内容：创建了cvToQimg
        日期：2022/1/28
        人员：吕凯凯
'''
def cvToQimg(cvimg):
    shrink = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QtGui.QImage(shrink.data, shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return qimg
def pilTocv(pil):
    cv = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return cv
def cvTopil(cv):
    cv = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(cv)
    return pil
#######################tools########################
predict = Predict()


######################深度模型接口##################
#线程利用全局变量传参
'''
SOD预测线程
'''
class Thread_Classification(QThread):  # 线程1
    def __init__(self):
        super().__init__()

    def run(self):
        window.result = predict.classification_pre(window.img1,window.img2)
        predict.finish_flag = True

thread_classification = Thread_Classification()

'''
目标检测线程
'''
class Thread_Ojrectdetection(QThread):  # 线程2
    def __init__(self):
        super().__init__()

    def run(self):
        window.result = predict.objectdetection_pre(window.img1)
        predict.finish_flag = True
thread_objectdetection = Thread_Ojrectdetection()


'''
变化检测线程
'''
class Thread_ChangeDetection(QThread):  # 线程2
    def __init__(self):
        super().__init__()

    def run(self):
        window.result = predict.changedetection_pre(window.img1, window.img2)
        predict.finish_flag = True
thread_changedetection = Thread_ChangeDetection()
######################深度模型接口###########################





######################GUI###########################
'''
    把每一次预测抽象成一个类HistImageItem，其继承于QPushButton
属性：
    1：
        名字：type
        类型：str
        作用：用于标记元素的类型
    2：
        名字：result
        类型：Qimge
        作用：用于存储预测结果
    3：
        名字：input1,input2
        类型：Qimge
        作用：用于存储输入的双时相图片
方法：
    1：
        名字：__init__()
        返回值：无
        参数：
            1：itemType:表示构造时所希望的元素类型
            2：imgDir：加载的图片，其类型是Qimge
            3：centerWidge：元素的父级控件
        作用：构造元素的函数
    2：
        名字：contextMenuEvent
        返回值：无
        参数：evt,用于传递位置,其类型是Qaction
        作用：将按钮右击事件与弹出菜单（popMenue）联系起来
    3：
       名字：saveBack()
       返回值：无
       参数：无
       作用：右键保存事件的槽函数
log:
    1:
        内容：创建了HistImageItem
        日期：2022/1/28
        人员：吕凯凯
'''
class HistImageItem(QPushButton):
    def __init__(self, itemType, img, centerWidge):
        super().__init__(centerWidge)
        print("构造item")
        self.type = itemType
        self.result = None
        self.input1 = None
        self.input2 = None
        self.setStyleSheet("QPushButton:hover{"
                           "color:red;"
                           "border-color:black;"
                           "border-style:solid;"
                           "border-width:2px;"
                           "background-color:rgb(77,88,99);"
                           "}")

        if self.type == 'Classification' or self.type == 'ObjectDetection':
            self.setIconSize(QtCore.QSize(100, 100))
            # self.setIcon(QIcon(imgDir))
        else:
            self.setIconSize(QtCore.QSize(220, 100))

        icon = QIcon()
        icon.addPixmap(QPixmap.fromImage(img))
        self.setIcon(icon)
        self.clicked.connect(self.loadImage)

    def contextMenuEvent(self, evt):  # 连接菜单事件
        menu = QMenu(self)
        saveAction = QAction("保存", menu)
        saveAction.triggered.connect(self.saveBack)
        deleteAction = QAction("删除", menu)
        deleteAction.triggered.connect(lambda:self.deleteLater())
        exitAction = QAction("退出", menu)
        exitAction.triggered.connect(lambda: menu.destroy())
        menu.addAction(saveAction)
        menu.addAction(deleteAction)
        menu.addAction(exitAction)
        menu.addSeparator()
        menu.exec_(evt.globalPos())

    def saveBack(self):
        window.fileDialog.setLabelText(QFileDialog.Accept, '保存')
        if window.fileDialog.exec_() == QFileDialog.Accepted:
            saveDir = window.fileDialog.selectedFiles()[0] + '.jpg'
            self.result.save(saveDir, "JPG", 100)

    def setResult(self, result):
        self.result = result

    def setInput1(self, input1):
        self.input1 = input1

    def setInput2(self, input2):
        self.input2 = input2

    def loadImage(self):
        window.labelOutput.setPixmap(QtGui.QPixmap.fromImage(self.result).scaled(200, 200))
        window.labelInput1.setPixmap(QtGui.QPixmap.fromImage(self.input1).scaled(150, 150))
        if self.type == 'ChangeDetection':
            window.labelInput2.setPixmap(QtGui.QPixmap.fromImage(self.input2).scaled(150, 150))

'''
一些静态控件的创建，以及布局等设置。
'''
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.centralwidget)
        # ////////////////////////FrameHistory////////////////////////////////////
        self.frameHistory = QtWidgets.QFrame(self.centralwidget)
        self.frameHistory.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameHistory.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frameHistory)
        self.labelHistory = QtWidgets.QLabel(self.frameHistory)
        self.labelHistory.setTextFormat(QtCore.Qt.AutoText)
        self.verticalLayout_8.addWidget(self.labelHistory)

        self.stackWidget = QtWidgets.QStackedWidget(self.frameHistory)
        self.verticalLayout_8.addWidget(self.stackWidget)

        self.scrollAreaClassification = QtWidgets.QScrollArea(self.stackWidget)
        self.scrollAreaClassification.setWidgetResizable(True)
        self.scrollAreaClassificationWidgetContents = QtWidgets.QWidget(self.scrollAreaClassification)
        self.scrollAreaClassificationWidgetContents.setGeometry(QtCore.QRect(0, 0, 227, 608))
        self.layoutscrollAreaClassification = QtWidgets.QVBoxLayout(self.scrollAreaClassificationWidgetContents)
        self.scrollAreaClassification.setWidget(self.scrollAreaClassificationWidgetContents)
        self.stackWidget.addWidget(self.scrollAreaClassification)

        self.scrollAreaObjectDetc = QtWidgets.QScrollArea(self.stackWidget)
        self.scrollAreaObjectDetc.setWidgetResizable(True)
        self.scrollAreaObjectDetcWidgetContents = QtWidgets.QWidget(self.scrollAreaObjectDetc)
        self.scrollAreaObjectDetcWidgetContents.setGeometry(QtCore.QRect(0, 0, 227, 608))
        self.layoutscrollAreaObjectDetc = QtWidgets.QVBoxLayout(self.scrollAreaObjectDetcWidgetContents)
        self.scrollAreaObjectDetc.setWidget(self.scrollAreaObjectDetcWidgetContents)
        self.stackWidget.addWidget(self.scrollAreaObjectDetc)

        self.scrollAreaChangeDetc = QtWidgets.QScrollArea(self.stackWidget)
        self.scrollAreaChangeDetc.setWidgetResizable(True)
        self.scrollAreaChangeDetcWidgetContents = QtWidgets.QWidget(self.scrollAreaChangeDetc)
        self.scrollAreaChangeDetcWidgetContents.setGeometry(QtCore.QRect(0, 0, 227, 608))
        self.layoutscrollAreaChangeDetc = QtWidgets.QVBoxLayout(self.scrollAreaChangeDetcWidgetContents)
        self.scrollAreaChangeDetc.setWidget(self.scrollAreaChangeDetcWidgetContents)
        self.stackWidget.addWidget(self.scrollAreaChangeDetc)

        self.horizontalLayout_7.addWidget(self.frameHistory)
        # ////////////////////////Line5/////////////////////////////////////
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.line_5.setLineWidth(1)
        self.line_5.setMidLineWidth(0)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.horizontalLayout_7.addWidget(self.line_5)
        # ////////////////////////FrameMain1////////////////////////////////////
        self.frameMain1 = QtWidgets.QFrame(self.centralwidget)
        self.frameMain1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameMain1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frameMain1)
        self.labelInput1 = QtWidgets.QLabel(self.frameMain1)
        self.labelInput1.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.labelInput1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.labelInput1.setLineWidth(2)
        self.labelInput1.setMidLineWidth(2)
        self.labelInput1.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_2.addWidget(self.labelInput1)

        self.line_4 = QtWidgets.QFrame(self.frameMain1)
        self.line_4.setLineWidth(2)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_2.addWidget(self.line_4)

        self.frameInput1Button = QtWidgets.QFrame(self.frameMain1)
        self.frameInput1Button.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameInput1Button.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frameInput1Button)
        spacerItem = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)

        self.pushButtonLoad1 = QtWidgets.QPushButton(self.frameInput1Button)
        self.horizontalLayout.addWidget(self.pushButtonLoad1)

        spacerItem1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_2.addWidget(self.frameInput1Button)

        self.labelInput2 = QtWidgets.QLabel(self.frameMain1)
        self.labelInput2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.labelInput2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.labelInput2.setLineWidth(2)
        self.labelInput2.setMidLineWidth(2)
        self.labelInput2.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_2.addWidget(self.labelInput2)

        self.line_3 = QtWidgets.QFrame(self.frameMain1)
        self.line_3.setLineWidth(2)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_2.addWidget(self.line_3)
        self.frameInput2Button = QtWidgets.QFrame(self.frameMain1)
        self.frameInput2Button.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameInput2Button.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frameInput2Button)
        spacerItem3 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.pushButtonLoad2 = QtWidgets.QPushButton(self.frameInput2Button)
        self.horizontalLayout_3.addWidget(self.pushButtonLoad2)
        spacerItem4 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)

        self.verticalLayout_2.addWidget(self.frameInput2Button)
        self.horizontalLayout_7.addWidget(self.frameMain1)
        # //////////////////////////Line2//////////////////////////////////
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.line_2.setLineWidth(1)
        self.line_2.setMidLineWidth(0)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.horizontalLayout_7.addWidget(self.line_2)
        # //////////////////////////frameMain2//////////////////////////////////
        self.frameMain2 = QtWidgets.QFrame(self.centralwidget)
        self.frameMain2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameMain2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frameMain2)
        self.frameOutput = QtWidgets.QFrame(self.frameMain2)
        self.frameOutput.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameOutput.setFrameShadow(QtWidgets.QFrame.Raised)
        self.gridLayout = QtWidgets.QGridLayout(self.frameOutput)
        spacerItem6 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem6, 1, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem7, 0, 0, 1, 1)
        self.labelOutputText = QtWidgets.QLabel(self.frameOutput)
        self.labelOutputText.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.labelOutputText, 1, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 1, 2, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem9, 0, 2, 1, 1)

        self.labelOutput = QtWidgets.QLabel(self.frameOutput)
        self.labelOutput.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.labelOutput.setFrameShadow(QtWidgets.QFrame.Raised)
        self.labelOutput.setLineWidth(2)
        self.labelOutput.setMidLineWidth(2)
        self.gridLayout.addWidget(self.labelOutput, 0, 1, 1, 1)

        self.verticalLayout_6.addWidget(self.frameOutput)
        self.line_6 = QtWidgets.QFrame(self.frameMain2)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.line_6.setLineWidth(2)
        self.verticalLayout_6.addWidget(self.line_6)

        self.frameFun = QtWidgets.QFrame(self.frameMain2)
        self.frameFun.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameFun.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_6.addWidget(self.frameFun)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frameFun)
        self.labelStatus = QtWidgets.QLabel(self.frameFun)
        self.labelStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.horizontalLayout_4.addWidget(self.labelStatus)

        self.frameFunc = QtWidgets.QFrame(self.frameMain2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frameFunc)
        self.ratioClassification = QtWidgets.QRadioButton(self.frameFunc)
        self.ratioClassification.setChecked(1)
        self.ratioObjiectDetetion = QtWidgets.QRadioButton(self.frameFunc)
        self.ratioChangeDetection = QtWidgets.QRadioButton(self.frameFunc)
        self.verticalLayout_3.addWidget(self.ratioClassification)
        self.verticalLayout_3.addWidget(self.ratioObjiectDetetion)
        self.verticalLayout_3.addWidget(self.ratioChangeDetection)
        self.horizontalLayout_4.addWidget(self.frameFunc)

        self.frameStatus = QtWidgets.QFrame(self.frameMain2)
        self.frameStatus.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameStatus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frameStatus)
        self.line_7 = QtWidgets.QFrame(self.frameStatus)
        self.line_7.setLineWidth(2)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_4.addWidget(self.line_7)
        self.textBrowserCmd = QtWidgets.QTextBrowser(self.frameStatus)
        self.verticalLayout_4.addWidget(self.textBrowserCmd)

        self.line_8 = QtWidgets.QFrame(self.frameStatus)
        self.line_8.setLineWidth(2)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_4.addWidget(self.line_8)

        self.framePercent = QtWidgets.QFrame(self.frameStatus)
        self.framePercent.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePercent.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.framePercent)
        # self.labelPercent = QtWidgets.QLabel(self.framePercent)
        # self.horizontalLayout_2.addWidget(self.labelPercent)
        self.progressBar = QtWidgets.QProgressBar(self.framePercent)
        self.progressBar.setProperty("value", 0)
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.verticalLayout_4.addWidget(self.framePercent)
        self.verticalLayout_6.addWidget(self.frameStatus)

        self.line_8 = QtWidgets.QFrame(self.frameStatus)
        self.line_8.setLineWidth(2)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalLayout_4.addWidget(self.line_8)

        self.horizontalLayout_7.addWidget(self.frameMain2)
        MainWindow.setCentralWidget(self.centralwidget)
        # ///////////////////////////Menubar///////////////////////////////////
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1094, 23))
        self.menuBar.setDefaultUp(False)

        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.actionSaveall = QtWidgets.QAction(MainWindow)
        self.actionDelall = QtWidgets.QAction(MainWindow)
        self.menuFile.addAction(self.actionSaveall)
        self.menuFile.addAction(self.actionDelall)

        self.menuFun = QtWidgets.QMenu(self.menuBar)
        self.actionClassification = QtWidgets.QAction(MainWindow)
        self.actionObjectDetection = QtWidgets.QAction(MainWindow)
        self.actionChangeDetection = QtWidgets.QAction(MainWindow)
        self.menuFun.addAction(self.actionClassification)
        self.menuFun.addAction(self.actionObjectDetection)
        self.menuFun.addAction(self.actionChangeDetection)

        self.menuPatern = QtWidgets.QMenu(self.menuBar)
        self.actionbachsize = QtWidgets.QAction(MainWindow)
        self.actionwhole = QtWidgets.QAction(MainWindow)
        self.menuPatern.addAction(self.actionbachsize)
        self.menuPatern.addAction(self.actionwhole)
        self.menuAbout = QtWidgets.QMenu(self.menuBar)

        MainWindow.setMenuBar(self.menuBar)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuFun.menuAction())
        self.menuBar.addAction(self.menuPatern.menuAction())
        self.menuBar.addAction(self.menuAbout.menuAction())
        #///////////////////////FileDialog/////////////////////////
        self.fileDialog = QtWidgets.QFileDialog()
        self.fileDialog.resize(800, 400)
        self.fileDialog.setWindowTitle("保存为")
        self.fileDialog.setLabelText(QFileDialog.LookIn, '目录')
        self.fileDialog.setLabelText(QFileDialog.FileName, '文件名')
        self.fileDialog.setLabelText(QFileDialog.FileType, '文件类型')
        # self.fileDialog.setNameFilter('.jpg')
        self.fileDialog.setLabelText(QFileDialog.Reject, '取消')
        self.fileDialog.setOption(QFileDialog.DontUseNativeDialog)
        #/////////////MessageBox/////////////////////////////
        self.messbox = QMessageBox(self)
        self.messbox.setInformativeText('请先加载第一张图片')
        self.messbox.setWindowTitle('提示')

        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        #全部控件的文本设置
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "遥感影像智能解译系统"))
        self.labelHistory.setText(_translate("MainWindow", "历史记录"))
        self.pushButtonLoad1.setText(_translate("MainWindow", "    加载图片    "))
        self.pushButtonLoad2.setText(_translate("MainWindow", "    加载图片    "))
        self.labelOutputText.setText(_translate("MainWindow", "检测结果"))
        self.ratioClassification.setText(_translate("MainWindow", "地物分类"))
        self.ratioObjiectDetetion.setText(_translate("MainWindow", "目标检测"))
        self.ratioChangeDetection.setText(_translate("MainWindow", "变化检测"))
        self.textBrowserCmd.setText("<font color=red>请加载一张图片!</font>")
        # self.labelPercent.setText(_translate("MainWindow", "完成百分比："))
        self.menuFile.setTitle(_translate("MainWindow", "文件"))
        self.menuFun.setTitle(_translate("MainWindow", "功能切换"))
        self.menuPatern.setTitle(_translate("MainWindow", "模式切换"))
        self.menuAbout.setTitle(_translate("MainWindow", "关于"))
        self.actionClassification.setText(_translate("MainWindow", "地物分类"))
        self.actionObjectDetection.setText(_translate("MainWindow", "目标检测"))
        self.actionChangeDetection.setText(_translate("MainWindow", "变化检测"))
        self.actionSaveall.setText('保存全部')
        self.actionDelall.setText('删除全部')
        self.actionbachsize.setText(_translate("MainWindow", "批量"))
        self.actionwhole.setText(_translate("MainWindow", "整体"))
        #一些控件的styleSheet设置
        self.fileDialog.setStyleSheet(
                                            "QFileDialog{""background-color:rgb(160,160,160)""}"      
                                            "QObject{"
                                            "font-family: '微软雅黑';"
                                            "font-size: 14px;"
                                            "font-weight: bold;"
                                           "}")

        self.pushButtonLoad1.setStyleSheet("QPushButton:click{"
                                           "border-color:black;"
                                           "border-style:solid;""border-width:2px;"
                                           "background-color:rgb(77,88,99);"
                                           "}"
                                           "QPushButton{background-color:rgb(215,215,215);}"
                                           )
        self.pushButtonLoad2.setStyleSheet("QPushButton:click{"
                                           "border-color:black;"
                                           "border-style:solid;"
                                           "border-width:2px;"
                                           "background-color:rgb(77,88,99);"
                                           "}"
                                           "QPushButton{background-color:rgb(215,215,215);}"
                                           )
        self.setStyleSheet("QMainWindow{""background-color:rgb(160,160,160);""}"
                           "QObject{        ""font-family: '微软雅黑';"
                                            "font-size: 14px;"
                                            "font-weight: bold;""}"
                           "QRadioButton{ ""font-family:'微软雅黑';"
                           "font-size: 18px;"
                           "font-weight: bold;""}"
                           )
        self.messbox.setStyleSheet("QWidget{""background-color:rgb(160,160,160);""}"
                                   "QPushButton{""background-color:rgb(215,215,215);""}"
                           "QObject{        ""font-family: '微软雅黑';"
                           "font-size: 14px;"
                           "font-weight: bold;""}"
                           "QRadioButton{ ""font-family:'微软雅黑';"
                           "font-size: 18px;"
                           "font-weight: bold;""}"
                           )
        self.menuBar.setStyleSheet("QMenuBar{""background-color:rgb(200,200,200);""}")
        self.fileDialog.setStyleSheet("QFileDialog {""background-color:rgb(160,160,160);""}")
        #初始化一些控件的图片
        self.labelStatus.setPixmap(QtGui.QPixmap('Classification.jpg').scaled(80, 80))
        self.labelOutput.setPixmap(QtGui.QPixmap('outputInit.jpg').scaled(200, 200))
        self.labelInput2.setPixmap(QtGui.QPixmap('input2Init.jpg').scaled(150, 150))
        self.labelInput1.setPixmap(QtGui.QPixmap("input1Init.jpg").scaled(150, 150))

        self.setWindowIcon(QIcon('skyLogo.jpg'))
        self.fileDialog.setWindowIcon(QIcon('file.jpg'))
        self.messbox.setWindowIcon(QIcon('info.jpg'))
'''
属性：
    1：
        名字：loadNext
        类型：int
        作用：标志位：用于实现时相1,2图片的顺序加载
    2：
        名字：change1Img
        类型：Qimg
        作用：用于暂存时相1图片
    3：
        名字：change2Img
        类型：Qimg
        作用：用于暂存时相2图片
    4：
        名字：imageDir1
        类型：str
        作用：用于暂存时相1图片的地址
    5：
        名字：imageDir2
        类型：str
        作用:用于暂存时相2图片的地址
方法：
    1：
        名字：binding()
        参数：无
        作用：用于事件的绑定
    2：
       名字：actionClassificationBack(),actionObjectDetectionBack(),actionChangeDetectionBack
       参数：无
       作用：菜单(menuFun)的三个事件的相应槽函数，实现功能的切换。
    3：
        名字：frameInput1ButtonBack，frameInput2ButtonBack
        参数：无
        作用：图片加载按钮1,2的槽函数，用于实现图片加载以及预测结果的产出。
log:
    1:
        内容：创建了HistImageItem
        日期：2022/1/28
        人员：吕凯凯   
'''


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        print("初始化")
        self.setupUi(self)
        self.loadNext = 0
        self.img1 = None
        self.img2 = None
        self.result = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.step = 1
        self.count = 0
        self.count1 = 0

    def binding(self):

        self.actionSaveall.triggered.connect(self.actionSaveallBack)
        self.actionDelall.triggered.connect(self.actionDeleteallBack)

        # 功能切换单击事件
        self.actionClassification.triggered.connect(self.actionClassificationBack)
        self.actionObjectDetection.triggered.connect(self.actionObjectDetectionBack)
        self.actionChangeDetection.triggered.connect(self.actionChangeDetectionBack)
        # ratiobutton单击事件
        self.ratioClassification.clicked.connect(self.actionClassificationBack)
        self.ratioObjiectDetetion.clicked.connect(self.actionObjectDetectionBack)
        self.ratioChangeDetection.clicked.connect(self.actionChangeDetectionBack)
        # 加载图片按钮单击事件
        self.pushButtonLoad1.clicked.connect(self.frameInput1ButtonBack)
        self.pushButtonLoad2.clicked.connect(self.frameInput2ButtonBack)
    # 菜单事件槽函数

    def actionClassificationBack(self):
        self.labelStatus.setPixmap(QtGui.QPixmap('Classification.jpg').scaled(100, 100))
        predict.status = 'Classification'
        self.ratioClassification.setChecked(1)
        self.stackWidget.setCurrentIndex(0)
        self.textBrowserCmd.setText("<font color=red>请加载一张图片!</font>")
        self.loadNext = 0
        print('地物分类')

    def actionObjectDetectionBack(self):
        self.labelStatus.setText('目标检测')
        predict.status = 'ObjectDetection'
        self.labelStatus.setPixmap(QtGui.QPixmap('Objectdetection.jpg').scaled(100, 100))
        self.ratioObjiectDetetion.setChecked(1)
        self.stackWidget.setCurrentIndex(1)
        self.textBrowserCmd.setText("<font color=red>请加载一张图片!</font>")
        self.loadNext = 0
        print('目标检测')

    def actionChangeDetectionBack(self):
        self.labelStatus.setPixmap(QtGui.QPixmap('Changedetection.jpg').scaled(100, 100))
        predict.status = 'ChangeDetection'
        self.ratioChangeDetection.setChecked(1)
        self.stackWidget.setCurrentIndex(2)
        self.textBrowserCmd.setText("<font color=red>请加载一张图片!</font>")
        print('变化检测')

    def actionSaveallBack(self):
        print('保存全部')
        rootDir = None
        window.fileDialog.setLabelText(QFileDialog.Accept, '保存')
        window.fileDialog.setLabelText(QFileDialog.Accept, '保存')
        if window.fileDialog.exec_() == QFileDialog.Accepted:
            rootDir = window.fileDialog.selectedFiles()[0]

        os.mkdir(rootDir)
        if predict.status == 'Classification':
            for each in range(self.layoutscrollAreaClassification.count()):
                dir = rootDir + '/' + str(each)
                os.mkdir(dir)
                item = self.layoutscrollAreaClassification.itemAt(each)
                result = item.widget().result
                saveDir = dir + '/result.jpg'
                result.save(saveDir, "JPG", 100)
                input1 =  item.widget().input1
                saveDir = dir + '/input.jpg'
                input1.save(saveDir, "JPG", 100)

        if predict.status == 'ObjectDetection':
            for each in range(self.layoutscrollAreaObjectDetc.count()):
                dir = rootDir + '/' + str(each)
                os.mkdir(dir)
                item = self.layoutscrollAreaObjectDetc.itemAt(each)
                result = item.widget().result
                saveDir = dir + '/result.jpg'
                result.save(saveDir, "JPG", 100)
                input1 = item.widget().input1
                saveDir = dir + '/input.jpg'
                input1.save(saveDir, "JPG", 100)

        if predict.status == 'ChangeDetection':
            for each in range(self.layoutscrollAreaChangeDetc.count()):
                dir = rootDir + '/' + str(each)
                os.mkdir(dir)
                item = self.layoutscrollAreaChangeDetc.itemAt(each)
                result = item.widget().result
                saveDir = dir + '/result.jpg'
                result.save(saveDir, "JPG", 100)

                input1 = item.widget().input1
                saveDir = dir + '/input1.jpg'
                input1.save(saveDir, "JPG", 100)

                input2 = item.widget().input2
                saveDir = dir + '/input2.jpg'
                input2.save(saveDir, "JPG", 100)

    def actionDeleteallBack(self):
        if predict.status == 'Classification':
            for each in range(self.layoutscrollAreaClassification.count()):
                item = self.layoutscrollAreaClassification.itemAt(each)
                item.widget().deleteLater()

        if predict.status == 'ObjectDetection':
            for each in range(self.layoutscrollAreaObjectDetc.count()):
                item = self.layoutscrollAreaObjectDetc.itemAt(each)
                item.widget().deleteLater()

        if predict.status == 'ChangeDetection':
            for each in range(self.layoutscrollAreaChangeDetc.count()):
                item = self.layoutscrollAreaChangeDetc.itemAt(each)
                item.widget().deleteLater()

    def frameInput1ButtonBack(self):
        self.fileDialog.setLabelText(QFileDialog.Accept, '打开')
        if self.fileDialog.exec_() == QFileDialog.Accepted:
            imageDir = self.fileDialog.selectedFiles()[0]

            if predict.status == 'ChangeDetection':
                self.labelInput1.setPixmap(QtGui.QPixmap(imageDir).scaled(150, 150))
                self.img1 = cv2.imread(imageDir)
                self.textBrowserCmd.setText("<font color=red>请加载二张图片!</font>")
                self.loadNext = 1

            elif predict.status == 'Classification':
                mat = scipy.io.loadmat(imageDir)['salinas_corrected']
                self.img1 = mat

                #伪色彩
                Data = mat[:,:,1:4]
                Data = Data.astype(np.float32)
                for band in range(0,3):
                    Data[:, :, band] = (Data[:, :, band] - np.min(Data[:, :, band])) / (
                            np.max(Data[:, :, band]) - np.min(Data[:, :, band]))
                Data = Data*255
                Data = Data.astype(np.uint8)

                self.labelInput1.setPixmap(QtGui.QPixmap.fromImage(cvToQimg(Data)).scaled(150, 150))
                thread_classification.start()
                self.timer.start(2000)

            elif predict.status == 'ObjectDetection':
                self.labelInput1.setPixmap(QtGui.QPixmap(imageDir).scaled(150, 150))
                self.img1 = cv2.imread(imageDir)
                thread_objectdetection.start()
                self.timer.start(25)

    def frameInput2ButtonBack(self):
        if self.loadNext == 1:
            if self.fileDialog.exec_() == QFileDialog.Accepted:
                imageDir = self.fileDialog.selectedFiles()[0]
                self.labelInput2.setPixmap(QtGui.QPixmap(imageDir).scaled(150, 150))
                self.loadNext = 0
                self.img2 = cv2.imread(imageDir)

                # self.img1 = cv2.resize(self.img1, (300, 300))
                # self.img2 = cv2.resize(self.img2, (300, 300))
                thread_changedetection.start()
                self.timer.start(15)
        else:
            if predict.status == 'ChangeDetection':
                self.messbox.setInformativeText('请先加载第一张图片')
                self.messbox.exec_()
            else:
                self.messbox.setInformativeText('请先切换到变化检测模式')
                self.messbox.exec_()
            print("禁止")

    def timeout(self):
        # 动态刷新稍等
        strr = '请稍后'
        self.count = self.count + self.step
        if self.count > 100:
            self.count = 100

        self.count1 = self.count1 + 1
        self.count1 = self.count1 % 8
        for each in range(self.count1):
            strr = strr + '.'

        print('timeout')
        self.textBrowserCmd.setText(strr)
        self.progressBar.setProperty("value", self.count)

        #刷新结果
        if predict.finish_flag:
            print("检测完成")
            predict.finish_flag = False
            self.timer.stop()

            if predict.status == 'Classification':
                #伪色彩
                Data = self.img1[:, :, 1:4]
                Data = Data.astype(np.float32)
                for band in range(0, 3):
                    Data[:, :, band] = (Data[:, :, band] - np.min(Data[:, :, band])) / (
                            np.max(Data[:, :, band]) - np.min(Data[:, :, band]))
                Data = Data * 255
                Data = Data.astype(np.uint8)

                item = HistImageItem('Classification', cvToQimg(Data), self.scrollAreaClassification)
                item.setResult(cvToQimg(self.result))
                item.setInput1(cvToQimg(Data))

                self.labelOutput.setPixmap(QtGui.QPixmap.fromImage(cvToQimg(self.result)).scaled(200, 200))
                self.layoutscrollAreaClassification.addWidget(item)
                self.textBrowserCmd.setText("<font color=red>请加载下一张图片!</font>")
                self.count = 0

            if predict.status == 'ObjectDetection':
                print("目标检测检测完成")
                item = HistImageItem('ObjectDetection', cvToQimg(self.img1), self.scrollAreaObjectDetc)
                item.setResult(cvToQimg(self.result))
                item.setInput1(cvToQimg(self.img1))

                self.labelOutput.setPixmap(QtGui.QPixmap.fromImage(cvToQimg(self.result)).scaled(200, 200))
                self.layoutscrollAreaObjectDetc.addWidget(item)
                self.textBrowserCmd.setText("<font color=red>请加载下一张图片!</font>")
                self.count = 0

            if predict.status == 'ChangeDetection':
                print("变化检测完成")
                img_inter = cv2.imread('Intervel.jpg')
                forward = cv2.resize(self.img1, (300, 300))
                backward = cv2.resize(self.img2, (300, 300))
                img_concat = cv2.hconcat([forward, img_inter, backward])
                item = HistImageItem('ChangeDetection', cvToQimg(img_concat), self.scrollAreaChangeDetc)

                item.setResult(cvToQimg(self.result))
                item.setInput1(cvToQimg(self.img1))
                item.setInput2(cvToQimg(self.img2))

                self.labelOutput.setPixmap(QtGui.QPixmap.fromImage(cvToQimg(self.result)).scaled(200, 200))
                self.layoutscrollAreaChangeDetc.addWidget(item)
                self.textBrowserCmd.setText("<font color=red>请加载下一张图片!</font>")
                self.count = 0

######################GUI###########################



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.resize(800, 600)
    window.show()
    window.binding()
    sys.exit(app.exec_())
