from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL.GL import *
import cv2
from math import *
import time
import os
from py3_simulation import control
from py3_simulation.interface import RobotInterface
from py3_simulation.vision.token_locator import QrFinder
UP = 119
DOWN = 115
LEFT = 97
RIGHT = 100
mainwindow_class = uic.loadUiType("title.ui")[0]
quitbox_class = uic.loadUiType("quitbox.ui")[0]
drone_label_count = 2   #드론라벨카운트 2부터 +1씩

class quitBox(QMainWindow, quitbox_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def accept(self):
        exit()
    def reject(self):
        self.close()

class MainWindow(QMainWindow, mainwindow_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.quitbox = quitBox()
        self.drone_label = []
        self.inputbattery = 100
        self.getGOODS() #제품목록 불러오기
        self.showBettery()
        self.paintTempmap()     #임시 맵 튜

    #드론에서 받은 사진 파라매터로 메인화면에 출력
    def start_drone(self):
        interface = RobotInterface()
        qrfinder = QrFinder()
        interface.set_target(0.5, 1.45, 0.2, 0)  # default location

        state = 1
        target = 1
        direction = 0
        task = [UP, UP, RIGHT, RIGHT, DOWN, LEFT, LEFT, DOWN, RIGHT, RIGHT, 27]
        # task = [RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN, 27]
        cv2.namedWindow('masked_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('masked_image', 1, 1)
        while True:
            time.sleep(0.05)

            img = interface.get_image_from_camera()
            self.showVideo(img)

            value = qrfinder.find_code(img)

            try:
                x = value[0].data
                section, current, top, right, bottom, left = x.split(b'|')
            except:
                section, current, top, right, bottom, left = 0, 0, 0, 0, 0, 0
                pass

            if target == 0:
                print('[*] Illegal target')
                break

            if target == current:
                direction = 0

            v = 0.030
            if direction == 1:
                interface.move(0, 0, v, 0)
                continue
            elif direction == 2:
                interface.move(0, -v, 0, 0)
                continue
            elif direction == 3:
                interface.move(0, 0, -v, 0)
                continue
            elif direction == 4:
                interface.move(0, v, 0, 0)
                continue

            # ch = cv2.waitKey(5) & 0xFF
            if direction == 0:
                ch, task = task[0], task[1:]
            print(section, current, top, right, left, bottom, ch)
            if ch == UP:
                target = top
                direction = 1
            elif ch == RIGHT:
                target = right
                direction = 2
            elif ch == DOWN:
                target = bottom
                direction = 3
            elif ch == LEFT:
                target = left
                direction = 4
            elif ch == 27:
                break

        interface.stop()
        cv2.destroyAllWindows()
        self.drone_locate(1, 0)

    def showVideo(self, cvImage):
        qimg = QtGui.QImage(cvImage, cvImage.shape[1], cvImage.shape[0], cvImage.strides[0], QtGui.QImage.Format_Grayscale8)
        self.video_frame.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def temp(self):
        self.setBatteryGuage(70)


    #공장 맵 그리기
    def paintTempmap(self):
        x, y = 1, 1
        test_moves = ['R', 'R', 'R', 'U', 'L', 'L', 'L', 'U', 'R', 'R', 'R', 'N', 'L', 'L', 'L', 'F']

        #드론 움직임에따른 테이블 그리드 생성
        for direction in test_moves:
            x, y = self.getDroneMoving(x, y, direction)

        #마지막 공백주기
        self.tempMap.insertColumn(self.tempMap.columnCount())
        self.tempMap.insertRow(0)
        #위젯 크기에 맞춰 행렬 사이즈 조정
        i, j = 0, 0
        width = self.tempMap.width() / self.tempMap.columnCount() - 1
        height = self.tempMap.height() / self.tempMap.rowCount() - 1
        while i < self.tempMap.columnCount():
            self.tempMap.setColumnWidth(i, width)
            i += 1
        while j < self.tempMap.rowCount():
            self.tempMap.setRowHeight(j, height)
            j += 1
    def getDroneMoving(self, x, y, dir):
            if (dir == 'N'):
                y += 3
                while y > self.tempMap.rowCount() - 1:
                    self.tempMap.insertRow(0)
            elif (dir == 'S'):
                y -= 2
            elif (dir == 'E'):
                x += 2
            elif (dir == 'W'):
                x -= 2
            elif (dir == 'R'):
                self.addReck(x, y)
                x += 1
                while x > self.tempMap.columnCount() - 1:
                    self.tempMap.insertColumn(x)
            elif(dir == 'L'):
                self.addReck(x, y)
                x -= 1
            elif(dir == 'U' or dir == 'D' or dir == 'F'):
                self.addReck(x, y)

            return x, y
    def addReck(self, x, y):
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(182, 78, 255))
        brush.setStyle(QtCore.Qt.BDiagPattern)
        item.setBackground(brush)
        item.setFlags(QtCore.Qt.ItemIsEditable)
        t = self.tempMap.rowCount() - y - 1
        self.tempMap.setItem(t, x, item)
    def drone_locate(self, x, y):
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEditable)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/drone/drone.png"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        item.setIcon(icon)
        self.tempMap.setIconSize(QtCore.QSize(30, 30))
        t = self.tempMap.rowCount() - y - 1
        self.tempMap.setItem(t, x, item)

    #비젼으로 제품 파악 후, 제품이름에 따른 갯수 화면에 표시 --> 인풋값 '제품명' '갯수'
    def setCount_goods(self, itemName, number):
        number = '\t' + str(number)
        i = 0
        while i < self.tableWidget.rowCount():
            temp = self.tableWidget.item(i, 0)
            if(temp.text() == itemName):
                item = QtWidgets.QTableWidgetItem()
                item.setText(number)
                font = QtGui.QFont()
                font.setBold(True)
                font.setWeight(75)
                item.setFont(font)
                item.setFlags(QtCore.Qt.ItemIsEditable)
                brush = QtGui.QBrush(QtGui.QColor(78, 182, 255))
                brush.setStyle(QtCore.Qt.NoBrush)
                item.setForeground(brush)
                self.tableWidget.setItem(temp.row(), 1, item)
                break
            i = i+1

    #제품목록 불러오기
    def getGOODS(self):
        i = 0
        self.tableWidget.clear()
        for dirName in os.listdir("./goods"):
            if(i > 0):
                self.tableWidget.insertRow(i)
            #제품명 등록
            item = QtWidgets.QTableWidgetItem()
            item.setText(dirName)
            item.setFlags(QtCore.Qt.ItemIsEditable)
            brush = QtGui.QBrush(QtGui.QColor(78, 182, 255))
            brush.setStyle(QtCore.Qt.NoBrush)
            item.setForeground(brush)
            self.tableWidget.setItem(i, 0, item)
            #checkable 피하기 위한 갯수도 빈칸으로 등록
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEditable)
            self.tableWidget.setItem(i, 1, item)
            self.tableWidget.resizeColumnsToContents()
            i = i+1

    #드론 라벨 추가
    def add_drone(self):

        dronename = "drone_label" + str(drone_label_count)
        new_drone_label = QtWidgets.QPushButton(self.layoutWidget)
        new_drone_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        new_drone_label.setAutoFillBackground(False)
        new_drone_label.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/drone_label/drone_label_clicked.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon.addPixmap(QtGui.QPixmap(":/drone_label/drone_label.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        new_drone_label.setIcon(icon)
        new_drone_label.setIconSize(QtCore.QSize(200, 80))
        new_drone_label.setCheckable(True)
        new_drone_label.setChecked(True)
        new_drone_label.setAutoRepeat(False)
        new_drone_label.setAutoDefault(False)
        new_drone_label.setObjectName(dronename)

        self.drone_label.append(new_drone_label)
        count = len(self.drone_label) - 1
        self.verticalLayout_drone.removeWidget(self.drone_pluslabel)
        self.verticalLayout_drone.addWidget(self.drone_label[count])
        if (len(self.drone_label) < 4):
            self.verticalLayout_drone.addWidget(self.drone_pluslabel)
        self.dronegroup.addButton(self.drone_label[count])

    #배터리게이지 업데이트 --> 인풋값 0~100
    def setBatteryGuage(self, number):
        self.inputbattery = number
        self.batteryGuage.update()
    #배터리 그리기 메소드
    def showBettery(self):
        self.batteryGuage.initializeGL()
        self.batteryGuage.paintGL = self.paintBattery
    #배터리 대쉬보드 그리기 + 게이지 그리기함수 호출
    def paintBattery(self):
        bettery_in = self.inputbattery
        bettery_out = bettery_in * 0.7 + 50
        glClear(GL_COLOR_BUFFER_BIT)

        glEnable(GL_BLEND) #옵션 활성
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) #알파값 투명하게
        img = cv2.imread('guage.png', cv2.IMREAD_UNCHANGED)
        glDrawPixels(260, 258, GL_BGRA, GL_UNSIGNED_BYTE, img)
        glColor3f(192, 0, 0)

        self.Torus2d(0.63,0.8,bettery_out)
    #배터리 게이지 그리기
    def Torus2d(self, inner, outer, pts):
        glBegin(GL_QUAD_STRIP)
        i = 50
        while i <= pts:
            angle = i / pts * 2 * pi+10
            glVertex2f(inner * cos(angle), inner * sin(angle))
            glVertex2f(outer * cos(angle), outer * sin(angle))
            i = i + 1
        glEnd()

    def turnOffAWS(self):
        self.quitbox.setFixedSize(300, 100)
        self.quitbox.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.quitbox.show()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mWindow = MainWindow()
    mWindow.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
    mWindow.setFixedSize(1348, 848)
    mWindow.show()
    app.exec_()
