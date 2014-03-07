# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'radio.ui'
#
# Created: Tue Mar  4 01:19:03 2014
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QWidget
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(729, 451)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.exitButton = QtGui.QPushButton(self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(30, 390, 141, 32))
        self.exitButton.setObjectName(_fromUtf8("exitButton"))
        self.stereoWidthSlider = QtGui.QSlider(self.centralwidget)
        self.stereoWidthSlider.setGeometry(QtCore.QRect(90, 260, 81, 22))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stereoWidthSlider.sizePolicy().hasHeightForWidth())
        self.stereoWidthSlider.setSizePolicy(sizePolicy)
        self.stereoWidthSlider.setMaximum(100)
        self.stereoWidthSlider.setProperty("value", 50)
        self.stereoWidthSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stereoWidthSlider.setObjectName(_fromUtf8("stereoWidthSlider"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 141, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.freqSelect = QtGui.QDial(self.centralwidget)
        self.freqSelect.setGeometry(QtCore.QRect(-10, 50, 201, 151))
        self.freqSelect.setAutoFillBackground(False)
        self.freqSelect.setMinimum(881)
        self.freqSelect.setMaximum(1079)
        self.freqSelect.setSingleStep(2)
        self.freqSelect.setSliderPosition(907)
        self.freqSelect.setOrientation(QtCore.Qt.Horizontal)
        self.freqSelect.setInvertedAppearance(False)
        self.freqSelect.setWrapping(False)
        self.freqSelect.setNotchTarget(6.7)
        self.freqSelect.setNotchesVisible(True)
        self.freqSelect.setObjectName(_fromUtf8("freqSelect"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(210, 20, 56, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.curFreq = QtGui.QLabel(self.centralwidget)
        self.curFreq.setGeometry(QtCore.QRect(40, 210, 121, 16))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(18)
        font.setItalic(True)
        self.curFreq.setFont(font)
        self.curFreq.setObjectName(_fromUtf8("curFreq"))
#
#         self.spectrumPlot = Qwt5.QwtPlot(self.centralwidget)
#         self.spectrumPlot.setGeometry(QtCore.QRect(170, 40, 431, 200))
#         self.spectrumPlot.setObjectName(_fromUtf8("spectrumPlot"))
#
        self.plotFrame = QWidget(self.centralwidget)
        self.plotFrame.setGeometry(QtCore.QRect(170, 40, 431, 200))
        self.plotFrame.setObjectName(_fromUtf8("plotFrame"))

        self.signalMeter = Qwt5.QwtThermo(self.centralwidget)
        self.signalMeter.setGeometry(QtCore.QRect(660, 30, 51, 191))
        self.signalMeter.setAlarmEnabled(False)
        self.signalMeter.setFillColor(QtGui.QColor(51, 114, 193))
        self.signalMeter.setMaxValue(0.0)
        self.signalMeter.setMinValue(-40.0)
        self.signalMeter.setProperty("value", -35.0)
        self.signalMeter.setObjectName(_fromUtf8("signalMeter"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(575, 10, 121, 21))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.checkBox = QtGui.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(10, 250, 85, 18))
        #self.checkBox.setChecked(True)
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(90, 240, 81, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))

        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(200, 300, 521, 121))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.demodfilt = QtGui.QWidget()
        self.demodfilt.setObjectName(_fromUtf8("demodfilt"))
        self.demodFilterSize = QtGui.QSlider(self.demodfilt)
        self.demodFilterSize.setGeometry(QtCore.QRect(170, 50, 160, 22))
        self.demodFilterSize.setMinimum(1)
        self.demodFilterSize.setMaximum(61)
        self.demodFilterSize.setSingleStep(2)
        self.demodFilterSize.setProperty("value", 21)
        self.demodFilterSize.setOrientation(QtCore.Qt.Horizontal)
        self.demodFilterSize.setTickPosition(QtGui.QSlider.TicksBelow)
        self.demodFilterSize.setTickInterval(5)
        self.demodFilterSize.setObjectName(_fromUtf8("demodFilterSize"))
        self.label_6 = QtGui.QLabel(self.demodfilt)
        self.label_6.setGeometry(QtCore.QRect(170, 30, 56, 13))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.demodFiltSizeLabel = QtGui.QLabel(self.demodfilt)
        self.demodFiltSizeLabel.setGeometry(QtCore.QRect(210, 30, 56, 13))
        self.demodFiltSizeLabel.setObjectName(_fromUtf8("demodFiltSizeLabel"))
        self.demodFiltMedian = QtGui.QRadioButton(self.demodfilt)
        self.demodFiltMedian.setGeometry(QtCore.QRect(20, 10, 97, 18))
        self.demodFiltMedian.setChecked(True)
        self.demodFiltMedian.setObjectName(_fromUtf8("demodFiltMedian"))
        self.demodFiltLP = QtGui.QRadioButton(self.demodfilt)
        self.demodFiltLP.setGeometry(QtCore.QRect(20, 50, 97, 18))
        self.demodFiltLP.setObjectName(_fromUtf8("demodFiltLP"))
        self.tabWidget.addTab(self.demodfilt, _fromUtf8(""))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.audioFilterSizeSlider = QtGui.QSlider(self.tab)
        self.audioFilterSizeSlider.setGeometry(QtCore.QRect(180, 50, 160, 22))
        self.audioFilterSizeSlider.setAutoFillBackground(False)
        self.audioFilterSizeSlider.setMinimum(1)
        self.audioFilterSizeSlider.setMaximum(24)
        self.audioFilterSizeSlider.setProperty("value", 16)
        self.audioFilterSizeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.audioFilterSizeSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.audioFilterSizeSlider.setTickInterval(2)
        self.audioFilterSizeSlider.setObjectName(_fromUtf8("audioFilterSizeSlider"))
        self.label_5 = QtGui.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(180, 20, 56, 13))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.audioFilterSizeText = QtGui.QLabel(self.tab)
        self.audioFilterSizeText.setGeometry(QtCore.QRect(230, 20, 56, 13))
        self.audioFilterSizeText.setObjectName(_fromUtf8("audioFilterSizeText"))
        self.audioFilterActive = QtGui.QCheckBox(self.tab)
        self.audioFilterActive.setGeometry(QtCore.QRect(40, 20, 85, 18))
        self.audioFilterActive.setChecked(True)
        self.audioFilterActive.setObjectName(_fromUtf8("audioFilterActive"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.drawPlot = QtGui.QCheckBox(self.centralwidget)
        self.drawPlot.setGeometry(QtCore.QRect(270, 17, 85, 21))
        self.drawPlot.setChecked(True)
        self.drawPlot.setObjectName(_fromUtf8("drawPlot"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(610, 220, 111, 81))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
####___
        self.waterfallButton = QtGui.QRadioButton(self.groupBox)
        self.waterfallButton.setGeometry(QtCore.QRect(10, 30, 97, 18))
        self.waterfallButton.setObjectName(_fromUtf8("waterfallButton"))
        self.plotButton = QtGui.QRadioButton(self.groupBox)
        self.plotButton.setGeometry(QtCore.QRect(10, 50, 97, 18))
        self.waterfallButton.setChecked(True)
        self.plotButton.setObjectName(_fromUtf8("plotButton"))
####`````
        self.groupBox_2 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(210, 230, 391, 61))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.spectrum_overall = QtGui.QRadioButton(self.groupBox_2)
        self.spectrum_overall.setGeometry(QtCore.QRect(20, 30, 97, 18))
        self.spectrum_overall.setChecked(True)
        self.spectrum_overall.setObjectName(_fromUtf8("spectrum_overall"))
        self.spectrum_channel = QtGui.QRadioButton(self.groupBox_2)
        self.spectrum_channel.setGeometry(QtCore.QRect(110, 30, 97, 18))
        self.spectrum_channel.setObjectName(_fromUtf8("spectrum_channel"))
        self.spectrum_playing = QtGui.QRadioButton(self.groupBox_2)
        self.spectrum_playing.setGeometry(QtCore.QRect(200, 30, 97, 18))
        self.spectrum_playing.setObjectName(_fromUtf8("spectrum_playing"))
        self.spectrum_waveform = QtGui.QRadioButton(self.groupBox_2)
        self.spectrum_waveform.setGeometry(QtCore.QRect(290, 30, 97, 18))
        self.spectrum_waveform.setObjectName(_fromUtf8("spectrum_waveform"))
        self.groupBox_3 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 280, 181, 101))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
####____
        self.mainchannel = QtGui.QRadioButton(self.groupBox_3)
        self.mainchannel.setGeometry(QtCore.QRect(10, 20, 121, 31))
        self.mainchannel.setChecked(True)
        self.mainchannel.setObjectName(_fromUtf8("mainchannel"))
        self.subband1 = QtGui.QRadioButton(self.groupBox_3)
        self.subband1.setGeometry(QtCore.QRect(20, 50, 141, 18))
        self.subband1.setObjectName(_fromUtf8("subband1"))
        self.subband2 = QtGui.QRadioButton(self.groupBox_3)
        self.subband2.setGeometry(QtCore.QRect(20, 70, 141, 21))
        self.subband2.setObjectName(_fromUtf8("subband2"))
#####````````
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 729, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QObject.connect(self.exitButton, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.close)
        QtCore.QObject.connect(self.audioFilterSizeSlider, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.audioFilterSizeText.setNum)
        QtCore.QObject.connect(self.demodFilterSize, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.demodFiltSizeLabel.setNum)
        QtCore.QObject.connect(self.mainchannel, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.checkBox.setEnabled)
        #QtCore.QObject.connect(self.mainchannel, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.checkBox.setChecked)
        QtCore.QObject.connect(self.subband1, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.checkBox.setDisabled)
        QtCore.QObject.connect(self.subband2, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.checkBox.setDisabled)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.exitButton.setText(_translate("MainWindow", "Quit", None))
        self.label.setText(_translate("MainWindow", "Frequency Selection", None))
        self.label_2.setText(_translate("MainWindow", "Spectrum", None))
        self.curFreq.setText(_translate("MainWindow", "90.7 MHz", None))
        self.label_3.setText(_translate("MainWindow", "Signal Strength - dB", None))
        self.checkBox.setText(_translate("MainWindow", "Stereo", None))
        self.label_4.setText(_translate("MainWindow", "Stereo Width", None))
        self.label_6.setText(_translate("MainWindow", "Size:", None))
        self.demodFiltSizeLabel.setText(_translate("MainWindow", "21", None))
        self.demodFiltMedian.setText(_translate("MainWindow", "Median", None))
        self.demodFiltLP.setText(_translate("MainWindow", "Lowpass", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.demodfilt), _translate("MainWindow", "Demodulation Filter", None))
        self.label_5.setText(_translate("MainWindow", "Width:", None))
        self.audioFilterSizeText.setText(_translate("MainWindow", "16", None))
        self.audioFilterActive.setText(_translate("MainWindow", "On", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Audio Filter", None))
        self.drawPlot.setText(_translate("MainWindow", "Draw", None))
        self.groupBox.setTitle(_translate("MainWindow", "Plot Type", None))
        self.waterfallButton.setText(_translate("MainWindow", "Waterfall", None))
        self.plotButton.setText(_translate("MainWindow", "Plot", None))
        self.groupBox_2.setTitle(_translate("MainWindow", "Plot Source", None))
        self.spectrum_overall.setText(_translate("MainWindow", "Overall", None))
        self.spectrum_channel.setText(_translate("MainWindow", "Channel", None))
        self.spectrum_playing.setText(_translate("MainWindow", "Playing", None))
        self.spectrum_waveform.setText(_translate("MainWindow", "Waveform", None))
        self.groupBox_3.setTitle(_translate("MainWindow", "Audio Source", None))
        self.mainchannel.setText(_translate("MainWindow", "Main Channel", None))
        self.subband2.setText(_translate("MainWindow", "Subband Channel 2", None))
        self.subband1.setText(_translate("MainWindow", "Subband Channel 1", None))
        #self.actionSetFreq.setText(_translate("MainWindow", "setFreq", None))

from PyQt4 import Qwt5
