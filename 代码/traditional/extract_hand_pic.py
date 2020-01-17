import cv2
import numpy as np

def get_hand_pic(img):

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, c_r, c_b) = cv2.split(img_yuv)

    '''cv2.imshow("y",y)
    cv2.imshow("c r", c_r)
    cv2.imshow("c b", c_b)
    cv2.waitKey(0)'''

    c_r_blur = cv2.GaussianBlur(c_r, (5, 5), 0)
    _, bi_img = cv2.threshold(c_r_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_open = np.ones((5, 5), np.uint8)
    open_bi = cv2.morphologyEx(bi_img, cv2.MORPH_OPEN, kernel_open)

    return open_bi