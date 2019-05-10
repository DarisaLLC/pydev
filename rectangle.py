import unittest


def fix(box):
    if box[0] > box[2]:
        tmp = box[0]
        box[0] = box[2]
        box[2] = tmp

    if box[1] > box[3]:
        tmp = box[1]
        box[1] = box[3]
        box[3] = tmp

    return box


def same(b1, b2):
    bb1 = fix(b1)
    bb2 = fix(b2)
    return bb1 == bb2


def area(ba):
    if not len(ba) == 4:
        return -1
    bb = fix(ba)
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def overlaps(box1, box2):
    """
    Returns if the two boxes overlap


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    overlaps_touches: bool = inter_rect_x1 <= inter_rect_x2 and inter_rect_y1 <= inter_rect_y2
    return overlaps_touches


__all__ = ['TestBBoxes']


class TestBBoxes(unittest.TestCase):

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """
    def test_bboxes(self):
        self.test_same()
        self.test_fix()
        self.test_overlaps()

    def test_same(self):
        b0 = [15, 17, 10, 12]
        bb0 = [15, 17, 10, 12]
        ok = same(b0,bb0)
        self.assertTrue(ok)
        bb0[0] = 16
        ok = same(b0, bb0)
        self.assertTrue(not ok)

    def test_fix(self):
        b0 = [15, 17, 10, 12]
        b1 = [10, 12, 15, 17]
        bf = fix(b0)
        ok = same(bf,b1)
        self.assertTrue(ok)

    def test_overlaps(self):

        b1 = [10, 12, 15, 17]
        b2 = [17, 12, 22, 17]
        ok = overlaps(b1, b2)
        self.assertTrue(not ok)

        b3 = [13, 12, 16, 17]
        ok = overlaps(b1, b3)
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
