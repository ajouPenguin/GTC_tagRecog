from pyzbar.pyzbar import decode
#from PIL import Image
import cv2
x = cv2.imread('qrcode.png')
print x

print decode(x)
