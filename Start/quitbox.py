# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'quitbox.ui'
#
# Created by: PyQt5 UI code generator 5.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QuitBox(object):
    def setupUi(self, QuitBox):
        QuitBox.setObjectName("QuitBox")
        QuitBox.resize(337, 128)
        self.buttonBox = QtWidgets.QDialogButtonBox(QuitBox)
        self.buttonBox.setGeometry(QtCore.QRect(50, 20, 241, 81))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(QuitBox)
        self.buttonBox.accepted.connect(QuitBox.accept)
        self.buttonBox.rejected.connect(QuitBox.reject)
        QtCore.QMetaObject.connectSlotsByName(QuitBox)

    def retranslateUi(self, QuitBox):
        _translate = QtCore.QCoreApplication.translate
        QuitBox.setWindowTitle(_translate("QuitBox", "Dialog"))

