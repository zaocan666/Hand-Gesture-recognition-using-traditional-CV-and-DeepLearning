import cv2
import math

from cofig import COMP_THREHOLD

#数伸出的手指个数
def count_hand_number(bi_hand, hand_rgb, countour=None):
    #bi_hand: binary image of hand
    #countour: result of findCountour function
    if type(countour)==type(None):
        _, countour, _= cv2.findContours(bi_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area=-1
    for i in range(len(countour)):  # find the biggest contour (according to area)
        ar = cv2.contourArea(countour[i])
        if ar > max_area:
            max_area = ar
            max_index = i

    if max_area==-1:
        return 0, hand_rgb
    hand_contour = countour[max_index]

    conv_hull = cv2.convexHull(hand_contour, returnPoints=False)

    if len(conv_hull) > 1:
        conv_defect = cv2.convexityDefects(countour[max_index], conv_hull)
        if type(conv_defect) == type(None):
            return 0, hand_rgb

        count = 0
        for i in range(conv_defect.shape[0]):  # calculate the angle
            # from https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python
            s, e, f, d = conv_defect[i][0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                count += 1
                cv2.circle(hand_rgb, far, 8, [200, 80, 0], -1)

        if count>0:
            return min(count+1, 5), hand_rgb
        else:
            comp = compactness(hand_contour)
            if comp>COMP_THREHOLD:
                return 1, hand_rgb
            else:
                return 0, hand_rgb
    else:
        return 0, hand_rgb

def compactness(counter):
    area = cv2.contourArea(counter)
    length = cv2.arcLength(counter, True)
    return (length**2)/area
