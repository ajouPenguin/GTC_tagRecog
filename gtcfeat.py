import cv2
import skimage.data
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import numpy as np 
import os
import selectivesearch

# img = cv2.imread('messi5.jpg')
#     5 
#     6 res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#     7 
#     8 #OR
#     9 
#    10 height, width = img.shape[:2]
#    11 res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)


def getFeat(img, algorithm = 'histogram', masksize = (32, 32)):
    maskRows = masksize[0]
    maskCols = masksize[1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize = (maskRows, maskCols), interpolation=cv2.INTER_CUBIC)

    if algorithm == 'histogram' :
        row_feat = [ 0 ] * maskRows
        col_feat = [ 0 ] * maskCols
        for r in range(maskRows):
            for c in range(maskCols):
                row_feat[r] += img[r][c]
                col_feat[c] += img[r][c]
        for r in range(maskRows):
            row_feat[r] /= float(maskCols*255)
        for c in range(maskCols):
            col_feat[c] /= float(maskRows*255)

        return row_feat + col_feat 

    elif algorithm == 'lbp':
        lbp_feat = [0] * 256
        for r in range(1, maskRows-1):
            for c in range(1, maskCols-1):
                lbp_value = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0: 
                            continue
                        #lbp_value *= 2
                        if img[r+dr][c+dc] > img[r][c] :
                            lbp_value += 1
                lbp_feat[ lbp_value ] += 1

        total = (maskCols-1)*(maskRows-1)
        for i in range(256):
            lbp_feat[i] /= float(total)

        return lbp_feat

    elif algorithm == 'mct':
        lbp_feat = [0] * 1023
        for r in range(0, maskRows):
            for c in range(0, maskCols):
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if r+dr < 0 or c+dc < 0 or r+dr >= maskRows or c+dc >= maskCols:
                            continue
                        if img[r+dr][c+dc] > img[r][c] and r*maskRows+c < 1023:
                            lbp_feat[r*maskRows+c] += 1

        total = maskCols * maskRows
        for i in range(1023):
            lbp_feat[i] /= float(total)

        return lbp_feat

    elif algorithm == 'hog':
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)

        return np.resize(hog_img, -1)

    elif algorithm == 'surf':
        surf = cv2.fe
        pass

    return None

def getDistance(feat1, feat2, algorithm='ssd'):
    if algorithm == 'ssd':
        ssd = 0
        dim = len(feat1)
        for i in range(dim):
            dd = feat1[i] - feat2[i] 
            ssd += dd * dd
        return ssd
    
    return None

def isTargetPattern(img, masksize = (32, 32)):
    maskRows = maskSize[0]
    maskCols = maskSize[1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize = (maskRows, maskCols), interpolation=cv2.INTER_CUBIC)

if __name__ == "__main__":
    positivePath = os.getcwd() + '/true'  
    negativePath = os.getcwd() + '/false'

    for file in os.listdir(imagepath):
        if not file.endswith('.JPG') :
            continue
        
        img = cv2.imread(imagepath + '/' + file, cv2.IMREAD_COLOR)
    
        feat = getFeat(img)
        print(feat)
    
