import os
import cv2
from skimage import io
import selectivesearch
import gtcfeat as gtc
import xml.etree.ElementTree as et

def extractFeature(img):
    return gtc.getFeat(img, algorithm='hog')# + gtc.getFeat(img, algorithm='mct')

negativePath = os.getcwd() + '/data/images'

path = os.getcwd() + "/data/images/"
imgList = os.listdir(path)
xmlList = os.listdir(path + 'xml/')

cnt = 0
frameCnt = 0

for li in imgList:
    xmlName = li[0:-3] + "xml"
    try:
        xmlFile = et.parse(path + "xml/" + li)
        root = xmlFile.getroot()
    except:
        xmlFile = None
        root = None
    cv_img = cv2.imread(path + li, cv2.IMREAD_COLOR)
    sk_img = io.imread(path + li)
    frameCnt += 1
    print("%d frame" % frameCnt)


    # perform selective search (selective search from https://github.com/AlpacaDB/selectivesearch)
    img_lbl, regions = selectivesearch.selective_search(sk_img, scale=1, sigma=0.9, min_size=100)

    candidates = set()
    for r in regions:
        # # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue

        # distorted rects
        x, y, w, h = r['rect']

        if w < 10 or h < 10 or w > 100 or h > 100 or w / h > 2 or h / w > 2:
            continue

        # if w / h > 2.0 or h / w > 2.0:
        #     continue
        
        candidates.add(r['rect'])

    for x, y, w, h in candidates:
        x1 = x 
        x2 = x + w - 1
        y1 = y 
        y2 = y + h - 1
        chk = 0

        xmin = []
        xmax = []
        ymin = []
        ymax = []
        if root != None:
            for itr in root.iter("xmin"):
                xmin.append(itr.text)
            for itr in root.iter("xmax"):
                xmax.append(itr.text)
            for itr in root.iter("ymin"):
                ymin.append(itr.text)
            for itr in root.iter("ymax"):
                ymax.append(itr.text)
        else:
            xmin.append(0)
            xmax.append(0)
            ymin.append(0)
            ymax.append(0)

        i = 0
        while i < len(xmin):
            if x1 > int(xmin[i]) and x2 < int(xmax[i]) and y1 > int(ymin[i]) and y2 < int(ymax[i]):
                chk = 1
                break
            elif x1 < int(xmax[i]) and int(xmax[i]) < x2 and y1 < int(ymax[i]) and int(ymax[i]) < y2:
                chk = 1
                break
            elif x1 < int(xmin[i]) and int(xmin[i]) < x2 and y1 < int(ymax[i]) and int(ymax[i]) < y2:
                chk = 1
                break
            elif x1 < int(xmax[i]) and int(xmax[i]) < x2 and y1 < int(ymin[i]) and int(ymin[i]) < y2:
                chk = 1
                break
            elif x1 < int(xmin[i]) and int(xmin[i]) < x2 and y1 < int(ymin[i]) and int(ymin[i]) < y2:
                chk = 1
                break
            elif x1 < int(xmin[i]) and int(xmax[i]) < y2 and y1 < int(ymin[i]) and y2 > int(ymax[i]):
                chk = 1
                break
            i += 1

        if chk == 1:
            continue

        cropped = cv_img[y1:y2, x1:x2]
        feat = extractFeature(cropped)
        # save false data
        fileName = os.getcwd() + "/data/images/bg/" + str(cnt) + ".jpg"
        cnt += 1
        cv2.imwrite(fileName,cropped)
        print("success to save " + str(cnt) + ".jpg")
        if cnt > 22500:
            break
