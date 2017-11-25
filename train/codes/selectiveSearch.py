import sys
import cv2

def selective_search(img, opt='q'):
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(8);

    # read image
    im = img

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if (opt == 'f'):
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    elif (opt == 'q'):
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()

    return rects
