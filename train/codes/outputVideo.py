from regionProposal import processing
import cv2

def outputVideo(clf, nonFiltered):
    # loading video
    print('Load video')
    try:
        mpgFile = '../data/input.mpg'
        vidcap = cv2.VideoCapture(mpgFile)
        cnt = 0
        falseCnt = 0

        prevRect = []
        printed = []

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter('../output/output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (height, width))

        print(length, width, height, fps)

        while (True):
            if cnt % 2 == 0:
                 del prevRect[:]
            ret, image = vidcap.read()
            if ret == 0:
                out.release()
                break
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
                print(image)
                out.write(image)
                #break
                print('%d frame' % (cnt))
    except Exception as e:
         out.release()
         print('Write Error')
         print(e)

    return None
