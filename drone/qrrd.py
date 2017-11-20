import zbarlight as zbar
from PIL import Image
import cv2

def qrDecode():
    video = cv2.VideoCapture('input.mpg')
    
    while(True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image = Image.fromarray(gray)

        codes = zbar.scan_codes('qrcode',image)

        print(codes)
