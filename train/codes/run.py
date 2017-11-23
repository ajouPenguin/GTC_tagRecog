import os
import gtcfeat as gtc
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
import math
import brightChange
import loadDBFromPath
from outputVideo import outputVideo

totalTime = time.time()
positivePath = []
for i in range(5):
    positivePath.append('../data/true/' + str(i))
negativePath = '../data/false'
newBG = []
for i in range(24):
    newBG.append(negativePath + '/bg/' + str(i))

# Load data for learning
try:
    clf = joblib.load('../output/dump.pkl')
    print('Using trained model')
except:
    clf = SVC()
    print('Building new model')
    print('Load data')
    db = []
    cnt = 1
    for i in positivePath:
        db += loadDBFromPath(i, 1)
        print('True (' + str(cnt) + '/5)')
        cnt += 1

    print('False')
    db += loadDBFromPath(negativePath, -1)

    cnt = 1
    for i in newBG:
        db += loadDBFromPath(i, -1)
        print('Background (' + str(cnt) + '/24)')
        cnt += 1

    # Make trainset and classes
    print('Make train set')
    trainset = np.float32([data['feat'] for data in db])
    classes = np.array([data['class'] for data in db])

    t_learn = time.time()
    # Start learning
    print('Start learning')
    clf.fit(trainset, classes)
    joblib.dump(clf, '../output/dump.pkl')

    print('learning time : %s sec' % str(time.time() - t_learn))

    # Recall check
    trueDB = []
    print('Check ACC')
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

try:
    nonFiltered = joblib.load('../output/realImgDump.pkl')
    print('Using trained model with non-filtered ')
except:
    print('Make non-filtered image training file')
    db = []
    nonFiltered = SVC()

    print('Load DB')

    cnt = 1
    for i in positivePath:
        db += loadDBFromPath(i, 1, 0)
        print('True (' + str(cnt) + '/5)')
        cnt += 1

    db += loadDBFromPath(negativePath, -1, 0)
    print('False')

    cnt = 1
    for i in newBG:
        db += loadDBFromPath(i, -1, 0)
        print('Background (' + str(cnt) + '/24)')
        cnt += 1

    # Make trainset and classes
    print('Make train set')

    data = []
    cnt = 0
    for itr in db:
        if cnt % int(len(db) / 10) == 0:
            print(str(math.ceil(cnt * 100 / len(db))) + '%')
        cnt += 1

        tmp = itr['feat']
        tmp2 = []
        for y in tmp[0]:
            for x in y:
                tmp2.append(x)

        data.append(tmp2)

    trainset = np.float32(data)
    classes = np.array([itr['class'] for itr in db])

    t_learn = time.time()
    # Start learning
    print('Start learning')
    nonFiltered.fit(trainset, classes)
    joblib.dump(nonFiltered, '../output/realImgDump.pkl')

    print('learning time : %s sec' % str(time.time() - t_learn))

    # Recall check
    trueDB = []
    print('Check ACC')
    for itr in db:
        if itr['class'] == 1:
    	    trueDB.append(itr['feat'])
    tot = 0
    cnt = 0
    for itr in trueDB:
        tot += 1
        if nonFiltered.predict([itr]) == 1:
    	    cnt += 1

    print('%f' % (cnt / tot))

outputVideo(clf, nonFiltered)