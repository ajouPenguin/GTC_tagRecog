from pyzbar.pyzbar import decode
import cv2

x = cv2.imread('qrcode.png')

qr = decode(x)
i = qr[0]
d = i.data
assert(d == b'LEFT 4\nRIGHT 6\nUP 8\nDOWN 2')


print('TEST DONE')
