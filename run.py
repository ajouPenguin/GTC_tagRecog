import sys
import os
import cv2
from skimage import io
import skvideo.io as vio
import selectivesearch
import gtcfeat as gtc
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import random

def brightChange(data, val):
    if(val == 0):
        val = random.randrange(-50,50)

    x = 0
    y = 0

    while(y < len(data)):
        while(x < len(data[y])):
            data[y][x] = (data[y][x] + val) % 255
            for i in range(3):
                if data[y][x][i] < 0:
                    data[y][x][i] = 0
            x += 1
        y += 1

    return data

def extractFeature(img):
    return gtc.getFeat(img, algorithm='hog')# + gtc.getFeat(img, algorithm='mct')


def loadDBFromPath(path, classnum):
    db = []
    for file in os.listdir(path):
        if not file.upper().endswith('.JPG'):
            continue
        data = {}
        data['class'] = classnum
        img = cv2.imread(path + '/' + file, cv2.IMREAD_COLOR)
        data['feat'] = extractFeature(img)
        db.append(data)
        for i in range(29):
            data['feat'] = extractFeature(brightChange(img, 0))
            db.append(data)
            if classnum != 1 and i > 5:
                break

    return db


def processing(sk_img, cv_img):
    # perform selective search (selective search from https://github.com/AlpacaDB/selectivesearch)
    img_lbl, regions = selectivesearch.selective_search(sk_img, scale=1, sigma=0.9, min_size=100)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue

        # distorted rects
        x, y, w, h = r['rect']

        if w < 10 or h < 10 or w > 100 or h > 100 or w / h > 2 or h / w > 2:
            continue

        candidates.add(r['rect'])

    # draw rectangles on the original image
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    #ax.imshow(sk_img)
    ret = []

    for x, y, w, h in candidates:
        x1 = x
        x2 = x + w - 1
        y1 = y
        y2 = y + h - 1
        cropped = cv_img[y1:y2, x1:x2]
        feat = extractFeature(cropped)
        feat = [feat]
        np.reshape(feat, (-1, 1))
        pred = clf.predict(feat)
        ret.append([x1, x2, y1, y2, pred])

    return ret


positivePath = os.getcwd() + '/data/true'
negativePath = os.getcwd() + '/data/false'
negativeBGPath = negativePath + '/bg'


#Load data for learning
try:
    clf = joblib.load('dump.pkl')
    print('Using trained model')
except:
    clf = SVC()
    print('Building new model')
    print("Load data")
    db = []
    print("true data")
    db += loadDBFromPath(positivePath, 1)
    print("false data")
    db += loadDBFromPath(negativePath, -1)
    print("bg data")
    db += loadDBFromPath(negativeBGPath, -1)

    #Make trainset and classes
    print("Make train set")
    trainset = np.float32([data['feat'] for data in db])
    classes = np.array([data['class'] for data in db])

    #Start learning
    print("Start learning")
    clf.fit(trainset, classes)
    joblib.dump(clf, 'dump.pkl')

    # Recall check
    trueDB = []
    print("Check recall")
    for itr in db:
        if itr['class'] == 1:
            trueDB.append(itr['feat'])
    tot = 0
    cnt = 0
    for itr in trueDB:
        tot += 1
        if clf.predict([itr]) == 1:
            cnt += 1

    print('%f' % (cnt / tot))


# loading images
sk_img = None
cv_img = None
if len(sys.argv) > 1 :
    fname = sys.argv[1]
    sk_img = io.imread(fname)
    cv_img = cv2.imread(fname, cv2.IMREAD_COLOR)
else :
    #print('Failed to load images')
    #exit(-1)

    mpgFile = 'output.mpg'
    vidcap = cv2.VideoCapture(mpgFile)
    skvid = vio.vreader(mpgFile)
    cnt = 0

    #while(vidcap.isOpened()):

    out = cv2.VideoWriter("a.mpg", cv2.VideoWriter_fourcc(*'mpeg'), 30.0, (752, 480))
    for frame in skvid:
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
        ret, image = vidcap.read()
        # 캡쳐된 이미지를 저장하는 함수
        if(int(vidcap.get(1))):
            cnt += 1
            rect = processing(frame, image)
            for (x1, x2, y1, y2, pred) in rect:
                if pred == 1:
                    ec = (0, 0, 255)
                    lw = 3
                #else:
                #    ec = (255, 0, 0)
                #    lw = 1
                    cv2.rectangle(image, (x1, y1), (x2, y2), ec, lw)
            out.write(image)

            #if(cnt % 30 == 0):
            print('%d frame transformation complete' % (cnt))
        if ret == 0:
            break

            # cv2.imshow('frame', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break