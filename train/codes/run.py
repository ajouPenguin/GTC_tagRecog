import os
import gtcfeat as gtc
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
import math
from brightChange import brightChange
from loadDBFromPath import loadDBFromPath
from outputVideo import outputVideo

def train(dataPath):

    db = []
    labels = []

    try:
        cnt = 0
        fileList = os.listdir(dataPath)
        if fileList == None:
            print('No file in path')
            return None
        for f in fileList:
            pa = os.path.join(dataPath, f)
            #print(pa)
            #print (os.access(pa, os.R_OK))
            db += loadDBFromPath(pa, cnt)
            cnt += 1
            labels.append(f)
    except Exception as e:
        print('No files in path')
        print(e)
        return None

    # HOG feature data training
    try:
        clf = joblib.load('../output/dump.pkl')
        print('Using trained model')
    except:
        clf = SVC()
        print('Building new model')
        print('Load data')

        # Make trainset and classes
        print('Make train set')
        trainset = np.float32([data['feat'] for data in db])
        classes = np.array([data['class'] for data in db])

        t_learn = time.time()
        # Start learning
        print('Start learning')
        clf.fit(trainset, classes)
        joblib.dump(clf, '../output/dump.pkl')

    outputVideo(clf)#, nonFiltered)

if __name__ == '__main__':
    train('../data')



'''    # non-filtered data training
    try:

        nonFiltered = joblib.load('../output/realImgDump.pkl')
        print('Using trained model with non-filtered ')
    except:
        print('Make non-filtered image training file')
        db = []
        try:
           for f in fileList:
                pa = os.path.join(dataPath, f)
                db += loadDBFromPath(pa, cnt, 0)
        except Exception as e:
            print('No files in path')
            print(e)
            return None

        nonFiltered = SVC()

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
        classes = [np.array([itr['class'] for itr in db])]

        print(trainset)
        print(classes)

        t_learn = time.time()

        # Start learning
        print('Start learning')
        nonFiltered.fit(trainset, classes)
        joblib.dump(nonFiltered, '../output/realImgDump.pkl')'''
