from regionProposal import processing
import cv2

def outputVideo(clf, nonFiltered):
    # loading video
    print('Load video')
    mpgFile = '../data/input.mpg'
    vidcap = cv2.VideoCapture(mpgFile)
    cnt = 0
    falseCnt = 0

    prevRect = []
    printed = []
    out = cv2.VideoWriter('../output/output.mpg', cv2.VideoWriter_fourcc(*'mpeg'), 30.0, (752, 480))
    while (True):
        if cnt % 2 == 0:
            del prevRect[:]
        ret, image = vidcap.read()
        if (int(vidcap.get(1))):
            numOfRect = 0
            rect = processing(image, clf, nonFiltered)

            for (x1, x2, y1, y2, pred) in rect:
                numOfRect += 1
                if pred == 1:
                    if cnt % 2 == 0:
                        temp = [x1, x2, y1, y2]
                        prevRect.append(temp)
                        for (x1, x2, y1, y2) in printed:
                            ec = (0, 0, 255)
                            lw = 3
                            cv2.rectangle(image, (x1, y1), (x2, y2), ec, lw)
                    else:
                        del printed[:]
                        for (px1, px2, py1, py2) in prevRect:
                            if abs(px1 - x1) < 60 and abs(px2 - x2) < 60 and abs(py1 - y1) < 60 and abs(py2 - y2) < 60:
                                ec = (0, 0, 255)
                                lw = 3
                                cv2.rectangle(image, (x1, y1), (x2, y2), ec, lw)
                                printed.append([x1, x2, y1, y2])
                                break
            cnt += 1
            out.write(image)
            print('%d frame' % (cnt))

        if ret == 0:
            break
