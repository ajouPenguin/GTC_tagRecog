import sys
import os
import cv2
import selectiveSearch as ss
import gtcfeat as gtc
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import random
import time

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
        if classnum == 1:
            for i in range(15):
                data['feat'] = extractFeature(brightChange(img, 0))
                db.append(data)

    return db


def processing(cv_img):
    # perform selective search (selective search from https://github.com/AlpacaDB/selectivesearch)
    rect = ss.selective_search(cv_img, opt = 'f')

    candidates = set()
    cnt = 0
    for r in rect:
        # excluding same rectangle (with different segments)
        x, y, w, h = r

        tmp = (x, y, w, h)

        if tmp in candidates:
            continue

        # distorted rects

        if w < 10 or h < 10 or w > 100 or h > 100 or w / h > 2 or h / w > 2:
            continue
        cnt += 1
        candidates.add(tmp)
    
    #print("Num of candidates:" + str(cnt))
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
positivePath2 = os.getcwd() + '/data/true2'
negativePath = os.getcwd() + '/data/false'
totalTime = time.time()

newBG = []
for i in range(15):
    newBG.append(negativePath + '/bg/' + str(i))

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
    print("true data")
    db += loadDBFromPath(positivePath2, 1)
    print("false data")
    db += loadDBFromPath(negativePath, -1)
    print("bg data")
    for i in newBG:
        db += loadDBFromPath(i, -1)

    #Make trainset and classes
    print("Make train set")
    trainset = np.float32([data['feat'] for data in db])
    classes = np.array([data['class'] for data in db])

    t_learn = time.time()
    #Start learning
    print("Start learning")
    clf.fit(trainset, classes)
    joblib.dump(clf, 'dump.pkl')

    print("learning time : %s sec" % str(time.time()-t_learn))

    # Recall check
    trueDB = []
    print("Check ACC")
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
cv_img = None
if len(sys.argv) > 1 :
    fname = sys.argv[1]
    cv_img = cv2.imread(fname, cv2.IMREAD_COLOR)
else :
    t_output = time.time()
    #print('Failed to load images')
    #exit(-1)
    mpgFile = 'output.mpg'
    vidcap = cv2.VideoCapture(mpgFile)
    cnt = 0
    falseCnt = 0

    #while(vidcap.isOpened()):

    out = cv2.VideoWriter("a.mpg", cv2.VideoWriter_fourcc(*'mpeg'), 30.0, (752, 480))
    while(True) :
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
        fps = time.time()
        ret, image = vidcap.read()
        # 캡쳐된 이미지를 저장하는 함수
        if(int(vidcap.get(1))):
            cnt += 1
            numOfRect = 0
            rect = processing(image)
            for (x1, x2, y1, y2, pred) in rect:
                numOfRect += 1
                if pred == 1:
                    #ec = (0, 0, 255)
                    #lw = 3
                    imgName = './catch/fa' + str(falseCnt) + '.jpg'
                    cv2.imwrite(imgName, image[y1:y2, x1:x2])
                    falseCnt += 1
                #else:
                    #ec = (255, 0, 0)
                    #lw = 1
                    #cv2.rectangle(image, (x1, y1), (x2, y2), ec, lw)
            out.write(image)
            #cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #if(cnt % 30 == 0):
            print('%d frame' % (cnt))

            fps = time.time() - fps
            print(str(fps) + "sec")

        if ret == 0:
            break

            # cv2.imshow('frame', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    print("Output time : %s sec", str(time.time() - t_output))

print("Total time : %s sec", str(time.time() - totalTime))
