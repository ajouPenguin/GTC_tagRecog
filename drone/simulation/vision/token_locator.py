# !/usr/bin/env python

from pyzbar.pyzbar import decode
import cv2
import numpy as np

from itertools import tee
import binascii
import math


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def nothing(kargs=[]):
    pass

class QrFinder():
    cap = None
    img = None

    def __init__(self):
        self.cap = None
        self.corrected = np.zeros((100, 100), np.uint8)  # image with corrected perspective

        cv2.namedWindow('edge')
        cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
        cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

        self.center = [0, 0]
        self.size = [0, 0]

        self.video = None
        self.width, self.height = [0, 0]



    def try_to_decode(self, candidate, source, target):
        epsilon = 0.01 * cv2.arcLength(candidate, True)
        approx = cv2.approxPolyDP(candidate, epsilon, True)

        while len(approx) > 4:
            epsilon *= 1.01
            approx = cv2.approxPolyDP(candidate, epsilon, True)

        center = sum(approx) / 4

        topleft = None
        topright = None
        bottomleft = None
        bottomright = None

        for i in approx:
            if i[0][0] < center[0][0] and i[0][1] < center[0][1]:
                topleft = i
            elif i[0][0] > center[0][0] and i[0][1] < center[0][1]:
                topright = i
            elif i[0][0] < center[0][0] and i[0][1] > center[0][1]:
                bottomleft = i
            elif i[0][0] > center[0][0] and i[0][1] > center[0][1]:
                bottomright = i

        self.center = center[0]
        self.size = [topright[0][0] - topleft[0][0], bottomright[0][1] - topright[0][1]]

        x, y, w, h = cv2.boundingRect(candidate)
        ddd = target[y: y + h, x: x + w]
        cv2.namedWindow('masked_image')
        cv2.imshow('masked_image', ddd)

        return decode(ddd)


    def find_code(self, img):
        h, w = img.shape[:2]

        gray=img
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        vis /= 2


        vis2 = np.zeros((h, w), np.uint8)
        vis2[edge != 0] = 255

        _, contours0, hierarchy = cv2.findContours(vis2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

        selected = []
        # [Next, Previous, First_Child, Parent]
        if np.all(hierarchy != None): # TEMP: Modified
            for c, h in zip(contours, hierarchy[0]):
                if h[0] == -1 and h[1] == -1:
                    kid = h[2]
                    if kid != -1:
                        kidh = hierarchy[0][kid]
                        if kidh[0] == -1 and kidh[1] == -1:
                            selected.append(c)
        result = []
        for candidate in selected:

            try:
                result = self.try_to_decode(candidate, gray, vis)
            except Exception as e:
                print(e)
            if result != []:
                break

        if self.video == None:
            height, width, layers =  vis.shape
            self.video = cv2.VideoWriter('video.mpg', cv2.VideoWriter_fourcc(*"mpeg"), 30.0, (width, height))

        cv2.imshow('contours', vis)
        self.video.write(vis)

        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            self.video.release()
            exit()

        return result

if __name__ == '__main__':
    finder = QrFinder()
