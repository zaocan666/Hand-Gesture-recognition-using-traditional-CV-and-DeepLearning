import cv2
import numpy as np
from hand_number import count_hand_number
from cofig import Area_x_start,Area_x_end,Area_y_start,BI_THRESHOLD,BLUR_SIZE

def get_num_mask(pic, mask):
    kernel_open = np.ones((5, 5), np.uint8)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((10, 10), np.uint8)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel_close)

    hand = cv2.bitwise_and(pic, pic, mask=closing_mask)
    hand = hand[int(Area_x_start * pic.shape[0]):int(Area_x_end * pic.shape[0]), int(Area_y_start * pic.shape[1]):]
    hand_g = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    hand_g = cv2.GaussianBlur(hand_g, (BLUR_SIZE, BLUR_SIZE), 0)

    _, bi_hand = cv2.threshold(hand_g, BI_THRESHOLD, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(bi_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hand, contours, -1, (0, 255, 0), 5)

    count, hand_rgb = count_hand_number(bi_hand, hand, countour=contours)

    return count, hand_rgb

def main():
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    START=False

    while camera.isOpened():
        _, pic = camera.read()
        pic = cv2.flip(pic, 1)
        cv2.imshow('pic', pic)
        cv2.rectangle(pic, (int(Area_y_start*pic.shape[1]),int(Area_x_start*pic.shape[0])), (pic.shape[1], int(Area_x_end*pic.shape[0])), (0,255,0),4)
        cv2.imshow('pic', pic)

        key = cv2.waitKey(8)
        if key==ord('m') or key==ord('M'): #m or M
            print("start")
            MOG2 = cv2.createBackgroundSubtractorMOG2(0,50)
            START=True
        elif key==ord('r') or key==ord('R'):
            cv2.destroyWindow('hand')
            MOG2=None
            START=False

        if START==True:
            mask = MOG2.apply(pic,learningRate=0)

            count, hand_rgb = get_num_mask(pic, mask)
            print(count)

            cv2.imshow('hand', hand_rgb)

if __name__=='__main__':
    main()