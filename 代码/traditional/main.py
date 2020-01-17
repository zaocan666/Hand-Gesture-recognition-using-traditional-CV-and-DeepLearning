import cv2
from extract_hand_pic import get_hand_pic
from hand_number import count_hand_number

if __name__=='__main__':
    img = cv2.imread('hand.jpg')
    bi_hand = get_hand_pic(img)
    cv2.imshow('bi hand', bi_hand)
    num, hand_rgb = count_hand_number(bi_hand, img)
    print(num)
    cv2.imshow('hand rgb', hand_rgb)
    cv2.waitKey(0)