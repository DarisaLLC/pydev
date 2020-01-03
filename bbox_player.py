import numpy as np
import cv2
import time
import copy
import random
import random_color as rc


class bboxSelector():

    def __init__(self, windowName):
        # cv2.namedWindow('image')
        cv2.setMouseCallback(windowName, self._drawBbox)
        self.ix = -1
        self.iy = -1
        self.drawing = -1
        self.stack = []
        self.windowName = windowName
        self.r, self.g, self.b = 97, 255, 98

    # rebuild should be used when the video is paused each time.
    def rebuild(self, image):

        cv2.setMouseCallback(self.windowName, self._drawBbox)
        self.ix = -1
        self.iy = -1
        self.drawing = -1
        # image is used to get original screenshot
        # cp_image is used to be the "not too original" image recording the result after each new bbox is added.
        # cp_step_image is used to show the image during drawing the bbox
        self.image = copy.deepcopy(image)
        self.cp_image = copy.deepcopy(image)
        self.cp_step_image = copy.deepcopy(image)
        # stack contains the (ix,iy,x,y) info for cut the sub img from the image at the end.
        self.stack = []

    # finish() is used to finish the operation during the pause process.
    def finish(self):
        nameStack = []
        imgStack = []
        for subImg in self.stack:
            tmp_str = "tmp_" + str(int(random.random() * 10000))
            nameStack.append(tmp_str)
            imgStack.append(self.image[subImg[1]:subImg[3], subImg[0]:subImg[2]])
            cv2.imshow(tmp_str, self.image[subImg[1]:subImg[3], subImg[0]:subImg[2]])
        return imgStack

    # cv2.setMouseCallback(self.windowName, None)
    def _drawBbox(self, event, x, y, flags, param):
        # self.cp_image = IMG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.cp_step_image = copy.deepcopy(self.cp_image)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.cp_step_image = copy.deepcopy(self.cp_image)
                cv2.rectangle(self.cp_step_image, (self.ix, self.iy), (x, y), (self.b, self.g, self.r), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            # here we use r.g.b inverseColor() to ensure the bbox color can be seen above any background color.
            b, g, r = np.mean(self.image[self.iy:y, self.ix:x, 0]), np.mean(
                self.image[self.iy:y, self.ix:x, 1]), np.mean(self.image[self.iy:y, self.ix:x, 2])
            # print(r,g,b)
            # r,g,b = rc.inverseColor(r,g,b)
            reid = "ReID:" + str(len(self.stack))
            r1, g1, b1, r2, g2, b2 = rc.contrastColor(r, g, b)
            # r1, g1, b1 = rc.inverseColor(r1, g1, b1)
            # r2, g2, b2 = rc.inverseColor(r2, g2, b2)
            cv2.rectangle(self.cp_image, (self.ix, self.iy), (x, y), (b2, g2, r2), 2)

            cv2.rectangle(self.cp_image, (self.ix - 1, y), (x + 1, y + 30), (b2, g2, r2), -1)
            cv2.putText(self.cp_image[y:(y + 30), (self.ix - 1):x], reid, (10, 26), cv2.FONT_HERSHEY_COMPLEX, 0.72,
                        (b1, g1, r1), 2)

            cv2.rectangle(self.cp_step_image, (self.ix, self.iy), (x, y), (b2, g2, r2), 2)

            cv2.rectangle(self.cp_step_image, (self.ix - 1, y), (x + 1, y + 30), (b2, g2, r2), -1)
            cv2.putText(self.cp_step_image[y:(y + 30), (self.ix - 1):x], reid, (10, 26), cv2.FONT_HERSHEY_COMPLEX, 0.72,
                        (b1, g1, r1), 2)

            self.stack.append([self.ix, self.iy, x, y])
            self.drawing = False


class ReIDPlayer():
    # init the player, with
    # source as the video file/ usb camera/ rtsp ...
    # windowName as the player window's name.
    def __init__(self, source, windowName):
        self.video = cv2.VideoCapture(source)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        # 15 is the time cost of each frame process, it should be auto calculate in future
        self.wait_time = int(1000 / self.fps) - 15
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar("Controller", windowName, 0, self.length, self._nothing)
        cv2.createTrackbar("Pause", windowName, 0, 1, self._nothing)
        cv2.setTrackbarPos("Pause", self.windowName, 1)

    def _nothing(self, others):
        pass

    def _close(self):
        self.video.release()
        cv2.destroyAllWindows()
        exit()

    def play(self):
        ret, frame = self.video.read()
        pos = 0
        bboxBoard = bboxSelector(self.windowName)
        while (True):
            cv2.waitKey(int(1000 / self.fps))
            v_pos = cv2.getTrackbarPos("Controller", self.windowName)
            if pos != v_pos:
                pos = v_pos
                self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
            else:
                pauseStatus = cv2.getTrackbarPos("Pause", self.windowName)
                if pauseStatus == 1:
                    pos += 1
                    cv2.setTrackbarPos("Controller", self.windowName, pos)
                    ret, frame = self.video.read()
                else:
                    bboxBoard.rebuild(frame)
                    while (cv2.getTrackbarPos("Pause", self.windowName) == 0):
                        cv2.imshow(self.windowName, bboxBoard.cp_step_image)
                        k = cv2.waitKey(1) & 0xff
                        if k == ord('s'):
                            print('you typed key s')
                            bboxBoard.finish()
                            break
                        elif k == ord('c'):
                            print("event bug fixed!")
                            bboxBoard.drawing = False
                        elif k == ord('p'):
                            print("Get screenshot successfully:")
                            print("The frameID is: " + str(v_pos))
                            cv2.imshow("screenshot", frame)
                        elif k == 27:
                            break
                        elif k == ord('q'):
                            self._close()

            cv2.imshow(self.windowName, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # time2 = time.time()
        # print(time2-time1)
        self._close()


def main():
    print("******" * 6)
    print("Press 'q' to quit!")
    print("When the video is paused, Press:")
    print("Press 'p' to get a full size screenshot.")
    print("Press 's' to get all the bbox image.")
    print("Press 'c' to get try to fix the bugs you might meet!")
    print("******" * 6)

    player = ReIDPlayer("img&video/init.mp4", "ReID")
    player.play()


if __name__ == "__main__":
    main()
