import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import pwiser


# returns r^^2
def ncv(image_a, image_b, options=None):
    res = cv2.matchTemplate(image_a, image_b, cv2.TM_CCOEFF_NORMED)
    # @todo just fetch[0,0]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val*max_val


if __name__ == '__main__':

    src_dir = sys.argv[1]
    assert(Path(src_dir).is_dir())
    p = Path(src_dir).glob('*.png')
    files = [x for x in p if x.is_file()]
    files.sort(key=lambda f: int(str(f.name).strip('frame').split('.')[0]))
    fItr = iter(files)
    q_size = 3
    buffer = []
    # create a pw unitary array
    # it sets the diagonal
    ssm = pwiser.getPairWiseArray((q_size,q_size))

    title = Path(src_dir).name
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)


    def getNextFile(iterator):
        try:
            item = next(iterator)
        except StopIteration:
            return None
        if not Path(item).is_file():
            return None
        return str(item)


    def exitIfNone(ff):
        if ff is None:
            print(('getNextFile', ' Failed'))
            sys.exit(1)

    # read one file to get the size
    # assumes all images have the same size
    file = getNextFile(fItr)
    exitIfNone(file)
    frame = cv2.imread(file, -1)
    shape = frame.shape
    buffer.append(frame)

    for pf in range(q_size - 1):
        file = getNextFile(fItr)
        exitIfNone(file)
        buffer.append(cv2.imread(file, -1))

    # create a deque for frame grabbing
    rb = deque(buffer)

    #fillup the ss with prefill comparisons
    for rr in range(q_size):
        for cc in range(rr+1, q_size):
            r = ncv(rb[rr],rb[cc])
            pwiser.setPairWiseArrayPair(ssm, rr, cc, r)

    ss = pwiser.getSelfSimilarity(ssm)
    sotime = []
    sotime.append(np.median(ss))

    '''
    On New Frame:
    queue append frame   
    queue popleft   oldest old new
    corr new against the rest and set 0, ....
    roll the ss matrix
    get similarity
    
    '''

    while (True):
        file = getNextFile(fItr)
        exitIfNone(file)
        frame = cv2.imread(file, -1)
        if frame is None: break
        ## put in FIFO
        rb.append(frame)
        rb.popleft()
        for ff in range(q_size - 1):
            r = ncv(frame, rb[ff])
            pwiser.setPairWiseArrayPair(ssm, 0, ff + 1, r)
        np.roll(ssm, (q_size - 1) * q_size)
        ss = pwiser.getSelfSimilarity(ssm)
        ssv = np.median(ss)
        print((ssv))
        sotime.append(ssv)
        plt.pause(0.1)
        cv2.imshow(title, frame)
        cv2.waitKey(1)

    plt.plot(sotime)
    plt.show()
