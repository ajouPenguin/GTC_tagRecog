import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import skimage
from skimage import io
import skimage.data
import selectivesearch
import gtcfeat as gtc
import numpy as np
from sklearn.svm import SVC


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
    return db

positivePath = os.getcwd() + '/data/true'
negativePath = os.getcwd() + '/data/false'
#negativeBGPath = negativePath + '/bg'

#Load data for learning
db = []
db += loadDBFromPath(positivePath, 1)
db += loadDBFromPath(negativePath, -1)
#db += loadDBFromPath(negativeBGPath, -1)
chk = 0

#Make trainset and classes
fileList = os.listdir(os.getcwd())
trainset = np.float32([data['feat'] for data in db])
classes = np.array([data['class'] for data in db])

#Start learning
if chk == 0:
    chk = 1
    clf = SVC()
    #Check that train data is in 'cwd'
    dataChk = 0
    for item in fileList:
        if item.find('svm_data.dat') is not -1:
            pass

    #No data in 'cwd'
    if dataChk == 0:
        clf.fit(trainset, classes)


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
    print('Training Error check')
    ratio = 0
    cnt = 0
    for data in db:
        pred = clf.predict([data['feat']])
        if pred == data['class']:
            ratio += 1
        cnt += 1

    ratio /= cnt
    print(ratio)
    exit(1)



# perform selective search (selective search from https://github.com/AlpacaDB/selectivesearch)
img_lbl, regions = selectivesearch.selective_search(sk_img, scale=500, sigma=0.9, min_size=20)

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
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(sk_img)

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

    if pred[0] == 1:
        ec = 'blue'
        lw = 3
    else:
        ec = 'red'
        lw = 1

    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor=ec, linewidth=lw)

    ax.add_patch(rect)
plt.show()
