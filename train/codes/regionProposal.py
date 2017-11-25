import selectiveSearch as ss
import gtcfeat as gtc
import numpy as np

# extract feature(default filter is hog)
def extractFeature(img):
    return gtc.getFeat(img, algorithm='hog')

# Perform selective search and return candidates
def processing(cv_img, clf):#, nonFiltered):
    # perform selective search
    rect = ss.selective_search(cv_img, opt='f')

    candidates = set()
    cnt = 0
    for r in rect:
        # excluding same rectangle (with different segments)
        x, y, w, h = r

        tmp = (x, y, w, h)

        if tmp in candidates:
            continue

        # distorted rects

        if w < 30 or h < 30 or w > 70 or h > 70 or w / h > 1.7 or h / w > 1.7:
            continue
        cnt += 1
        candidates.add(tmp)

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
#        if pred == 1:
#            feat = []
#            cropped = cv2.resize(cropped, (50, 50))
#            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#            for y in cropped:
#                for x in y:
#                    feat.append(x)
#            feat = [feat]
#            pred = nonFiltered.predict(feat)
        ret.append([x1, x2, y1, y2, pred])

    return ret
