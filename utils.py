import cv2
import numpy as np
import itertools
import math



def drawString(img, text):
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
def circleContainsPoint(center, radius_2, point):
    dx = center[0] - point[0]
    dy = center[1] - point[1]
    dd = dy * dy + dx * dx
    return dd < radius_2


def circleContainsLine(center, radius_2, line):
    x1, y1, x2, y2 = line
    if not circleContainsPoint(center, radius_2, (x1, y1)): return False
    if not circleContainsPoint(center, radius_2, (x2, y2)): return False
    return True


def circleContainsBoundingRect(center, radius_2, bb):
    if not circleContainsPoint(center, radius_2, (bb[0], bb[1])): return False
    if not circleContainsPoint(center, radius_2, (bb[0] + bb[2], bb[1])): return False
    if not circleContainsPoint(center, radius_2, (bb[0] + bb[2], bb[1] + bb[3])): return False
    if not circleContainsPoint(center, radius_2, (bb[0], bb[1] + bb[3])): return False
    return True


def draw_cross(img, center, color, d):
    cv2.line(img,
             (center[0] - d, center[1]), (center[0] + d, center[1]),
             color, 1, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0], center[1] - d), (center[0], center[1] + d),
             color, 1, cv2.LINE_AA, 0)



def find_all_intersections(lines):
    """Finds intersection points, if any, between all pairs of lines.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2), i.e. two points on the line.

    Returns:
        List of (x, y) intersection points. These are not unique.
    """
    intersection_points = []
    for line_a, line_b in itertools.combinations(lines, 2):
        point = find_intersection(line_a, line_b)
        if point is not None:
            intersection_points.append(point)
    return intersection_points


def find_intersection(line_a, line_b):
    """Finds intersection point between two lines, if any.

    Args:
        line_a: Vector of x1, y1, x2, y2, i.e. two points on the line.
        line_b: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Vector of x, y, the point of intersection. May be None for parallel
        or duplicate lines.
    """
    ax1, ay1, ax2, ay2 = np.array(line_a, dtype=np.float64)
    bx1, by1, bx2, by2 = np.array(line_b, dtype=np.float64)
    denominator = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)
    if denominator == 0:
        return None
    x0 = ((ax1 * ay2 - ay1 * ax2) * (bx1 - bx2) - (
            ax1 - ax2) * (bx1 * by2 - by1 * bx2)) / denominator
    y0 = ((ax1 * ay2 - ay1 * ax2) * (by1 - by2) - (
            ay1 - ay2) * (bx1 * by2 - by1 * bx2)) / denominator
    return x0, y0


def point_to_line_dist(point, line):
    """Finds euclidean distance between a point and a line.

    Args:
        point: Tuple (x, y) point.
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, distance.
    """
    px, py = np.array(point, dtype=np.float64)
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    nominator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    return nominator / point_to_point_dist((x1, y1), (x2, y2))


def point_to_point_dist(point_a, point_b):
    """Finds euclidean distance between two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Float, distance.
    """
    x1, y1 = np.array(point_a, dtype=np.float64)
    x2, y2 = np.array(point_b, dtype=np.float64)
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def find_nearest_point(point_a, points):
    """Finds the point among a set of points nearest to a given point.

    Args:
        point_a: (x, y) point tuple.
        points: Iterable of (x, y) point tuples.

    Returns:
        Tuple of:
            An (x, y) point tuple.
            Its distance to point_a (float).
    """
    nearest_point = None
    nearest_distance = None
    for point in points:
        dist = point_to_point_dist(point_a, point)
        if nearest_point is None or dist < nearest_distance:
            nearest_distance = dist
            nearest_point = point
    return nearest_point, nearest_distance


def get_midpoint(point_a, point_b):
    """Finds the midpoint of two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Tuple of (x, y) midpoint.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_line_angle(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return math.degrees(radians) % 360


def get_line_slope(line):
    """Calculates the slope of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, slope value.

    Raises:
        ZeroDivisionError, if slope is undefined (vertical lines).
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    denom = x2 - x1
    if denom == 0:
        raise ZeroDivisionError
    return (y2 - y1) / denom


def get_line_length(line):
    """Calculates the length of a line segment.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, line length.
    """
    return point_to_point_dist((line[0], line[1]), (line[2], line[3]))


def find_bounding_points(points):
    """Finds the minimum points that bound the set of points.

    In 2d, this is the lower left and upper right corners of
    a rectangle that contains all the given points. Overlap is
    allowed.

    Args:
        points: List of points.

    Returns:
        Tuple of min and max bounding points tuples.
    """
    if not points:
        return None, None
    min_bounding_point = list(points[0])
    max_bounding_point = list(points[0])
    for point in points:
        for i, coord in enumerate(point):
            min_bounding_point[i] = min(min_bounding_point[i], coord)
            max_bounding_point[i] = max(max_bounding_point[i], coord)
    return min_bounding_point, max_bounding_point


def find_point_on_rect_border(rect, angle):
    """Finds the point on the border at the given angle from center.

    Args:
        rect: Tuple of two points, the lower left and upper right
            corners of the rectangle.
        angle: Angle, in degrees.

    Returns:
        A point tuple on the rectangle's border.
    """
    # Grow the rectangle into a square, and find the border point on that,
    # since it's easier.
    width = abs(rect[1][0] - rect[0][0])
    height = abs(rect[1][1] - rect[0][1])
    square_size = max(width, height)
    square = [rect[0], (rect[0][0] + square_size, rect[0][1] + square_size)]
    border_x, border_y = _find_point_on_square_border(square, angle)
    # Scale the result to find the corresponding point on the original rectangle.
    border_x = border_x * (width / square_size)
    border_y = border_y * (height / square_size)
    return int(round(border_x)), int(round(border_y))


def _find_point_on_square_border(square, angle):
    """Finds the point on the border at the given angle from center.

    Based on: https://stackoverflow.com/a/1343531

    Args:
        square: Tuple of two points, the lower left and upper right
            corners of the square.
        angle: Angle, in degrees.

    Returns:
        A point tuple on the square's border.
    """
    angle = math.radians(angle)
    width = abs(square[1][0] - square[0][0])
    height = abs(square[1][1] - square[0][1])
    assert width == height
    center_x = square[0][0] + width / 2
    center_y = square[0][1] + height / 2
    abs_cos_angle = abs(math.cos(angle))
    abs_sin_angle = abs(math.sin(angle))
    if width / 2 * abs_sin_angle <= height / 2 * abs_cos_angle:
        magnitude = width / 2 / abs_cos_angle
    else:
        magnitude = height / 2 / abs_sin_angle
    border_x = center_x + math.cos(angle) * magnitude
    border_y = center_y + math.sin(angle) * magnitude
    return int(round(border_x)), int(round(border_y))

