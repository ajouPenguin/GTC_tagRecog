from interface import RobotInterface
import time
import cv2
from vision.token_locator import QrFinder
import threading
UP = 119
DOWN = 115
LEFT = 97
RIGHT = 100

class PidController():
    def __init__(self, P=0.26 ,I=0.001 ,D=0.0):
        self.P = P
        self.I = I
        self.D = D
        self.cumulatedError = 0
        self.lastError = 0
        self.lasttime = time.time()

    def update(self, currentValue, targetValue):
        newtime = time.time()
        dt = newtime - self.lasttime
        self.lasttime = newtime

        error = targetValue - currentValue
        derivative = self.D * (error - self.lastError)
        proportional = error * self.P
        self.cumulatedError += error * dt
        #self.cumulatedError = min(max(-100, self.cumulatedError), 100)
        integral = self.I * self.cumulatedError
        self.lastError = error

        #print "P:", proportional, " I:", integral, " D:", derivative
        return derivative + proportional + integral

class Controller:
    def __init__(self):
        self.interface = RobotInterface()
        self.qrfinder = QrFinder()
        self.interface.set_target(0.5, 1.45, 0.2, 0) # default location

        self.pidx = PidController(0.0001, 0.0001, 0.0001)
        self.pidy = PidController(0.0002, 0.0001, 0.001)
        self.pidz = PidController(0.0001, 0, 0)

        self.state = 1
        self.target = 1
        self.direction = 0
        task = [UP, UP, RIGHT, RIGHT, DOWN, LEFT, LEFT, DOWN, RIGHT, RIGHT, 27]
        #task = [RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN, 27]
        while True:
            time.sleep(0.05)

            img = self.interface.get_image_from_camera()

            value = self.qrfinder.find_code(img)

            try:
                x = value[0].data
                section, current, top, right, bottom, left = x.split(b'|')
            except:
                section, current, top, right, bottom, left = 0, 0, 0, 0, 0, 0
                pass

            if self.target == 0:
                print('[*] Illegal target')
                break

            if self.target == current:
                self.direction = 0

            v = 0.030
            if self.direction == 1:
                self.interface.move(0, 0, v, 0)
                continue
            elif self.direction == 2:
                self.interface.move(0, -v, 0, 0)
                continue
            elif self.direction == 3:
                self.interface.move(0, 0, -v, 0)
                continue
            elif self.direction == 4:
                self.interface.move(0, v, 0, 0)
                continue

            #ch = cv2.waitKey(5) & 0xFF
            if self.direction == 0:
                ch, task = task[0], task[1:]
            print(section, current, top, right, left, bottom, ch)
            if ch == UP:
                self.target = top
                self.direction = 1
            elif ch == RIGHT:
                self.target = right
                self.direction = 2
            elif ch == DOWN:
                self.target = bottom
                self.direction = 3
            elif ch == LEFT:
                self.target = left
                self.direction = 4
            elif ch == 27:
                break

        self.interface.stop()
        #cv2.destroyAllWindows()

    def control(self):
        targetx = 256
        targety = 256
        targetsize = 155

        dx = self.pidx.update(self.qrfinder.size[0], targetsize)
        dy = self.pidy.update(self.qrfinder.center[0], targetx)
        dz = self.pidz.update(self.qrfinder.center[1], targety)

        print((dx, dy, self.qrfinder.size))

        #distancia = 1 / (self.qrfinder.size[1] + 0.00001)

        self.interface.move(dx, dy, dz, 0)


if __name__ == '__main__':
    Controller()
