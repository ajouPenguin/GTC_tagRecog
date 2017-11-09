import zbarlight as zbar
from PIL import Image
import cv2

def main():
    image = cv2.imread('a.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = Image.fromarray(gray)

    codes = zbar.scan_codes('qrcode',image)

    print(codes)

if __name__ == "__main__":
    main()


