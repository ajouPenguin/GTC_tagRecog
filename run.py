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

#code start 
# loading astronaut image
sk_img = None
cv_img = None

if len(sys.argv) > 1 :
    fname = sys.argv[1]
    sk_img = io.imread(fname)
    cv_img = cv2.imread(fname, cv2.IMREAD_COLOR)
else :
    print('Failed to load images')
    exit(-1)


def extractFeature(img):
    return gtc.getFeat(img, algorithm = 'lbp')

def loadDBFromPath(path, classnum):
    db = []
    for file in os.listdir(path):
        if not file.upper().endswith('.JPG') :
            continue
        data = {}
        data['class'] = classnum
        img = cv2.imread(path + '/' + file, cv2.IMREAD_COLOR)
        data['feat'] = extractFeature(img)
        db.append(data)
    return db 

positivePath = os.getcwd() + '/data/true'  
negativePath = os.getcwd() + '/data/false'

db = []
db += loadDBFromPath(positivePath, 1)
db += loadDBFromPath(negativePath, 0)


# perform selective search
img_lbl, regions = selectivesearch.selective_search(sk_img, scale=500, sigma=0.9, min_size=10)

candidates = set()
for r in regions:
    # # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # # excluding regions smaller than 2000 pixels
    # if r['size'] <100  or r['size']  > 1000 :
    #     continue

    # distorted rects
    x, y, w, h = r['rect']

    if w < 10 or h < 10 or w > 100 or h > 100 or w / h > 2 or h / w > 2:
        continue


    # if w / h > 2.0 or h / w > 2.0:
    #     continue
    
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

    vote = [] 
    for data in db:
        dist = gtc.getDistance(feat, data['feat'])
        vote.append(  (dist, data['class']) ) 

    vote.sort()
    positive = 0
    negative = 0
    for i in range(3):
        if vote[i][1] == 1 :
            positive += 1 
        else:
            negative += 1 
    
    ec = 'blue'
    lw = 3
    if  negative >= positive:
        ec = 'red'
        lw = 1
    
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor=ec, linewidth=lw)
    
    ax.add_patch(rect)

plt.show()