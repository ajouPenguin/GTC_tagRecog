import os
import sys

list = os.listdir('./catch')

for li in list:
    try:
        isDir = int(li)
        if isDir < 20:
            continue
    except:
        pass
    try:
        fileNum = int(li[2:-4])
        folderNum = fileNum // 10000
    except:
        print("Error:%d" % fileNum)
    
    a = "mv ./catch/" + li + ' ./catch/' + str(folderNum)

    os.system(a)
