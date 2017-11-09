import cv2
import os

fList = os.listdir('./data/true')

for li in fList:
    if li.upper().endswith('.PNG'):
        img = cv2.imread('./data/true/' + li)
        fileName = "./data/true/" + li[0:-3] + 'jpg'
        os.remove('./data/true/' + li)
        cv2.imwrite(fileName, img)