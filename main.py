import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import random

def contact(contours):
    global stain_point
    for contour in contours:
        if contour['x'] <= stain_point[0] and contour['x'] + contour['w'] >= stain_point[0]:
            return True

def make_stain(img):
    global stain, stain_point
    #stain_imae = cv2.resize(stain, (0, 0), fx = 0.025, fy = 0.025, interpolation=cv2.INTER_LINEAR)

    stain_H,stain_W,c  = stain.shape

    height, width, channel = img.shape
    rand_Y = random.randint(0,height - stain_H)
    rand_X = random.randint(0,width - stain_W)
    stain_point[0] = rand_X
    stain_point[1] = rand_Y
    stain_point[2] = rand_X + stain_W
    stain_point[3]=  rand_Y + stain_H

    #img[rand_Y:rand_Y + stain_H, rand_X:rand_X + stain_W] = stain_imae


def trans(img):
    global start, stain, stain_point
    if start:
        make_stain(img)
        start = False
    img_ori = img
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    contours = max(contours, key=lambda x: cv2.contourArea(x))
#    cv2.drawContours(img_ori, [contours], -1, (255, 255, 0), 2)
#
#    x, y, w, h = cv2.boundingRect(contours)
#    print(x,y,w,h)
#    print("area : ",w*h,"ratio : ",w/h)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })


    MIN_AREA = 16000
    MIN_WIDTH, MIN_HEIGHT = 90, 90
    MIN_RATIO, MAX_RATIO = 0.45, 0.9

    possible_contours = []

    sum_height = 0
    sum_width = 0

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        sum_width += d['w']
        sum_height += d['h']

        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    for d in possible_contours:
        cv2.drawContours(img_ori, d['contour'], -1, (255, 255, 0))
        cv2.rectangle(img_ori, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255), thickness=2)
    #    mp_hands = mp.solutions.hands
#    mp_drawing = mp.solutions.drawing_utils
#
#    hands = mp_hands.Hands(
#        max_num_hands=2,
#        min_detection_confidence=0.5,
#        min_tracking_confidence=0.5)
    if contact(possible_contours):
        make_stain(img_ori)
    print(stain.shape)

    #img_ori[stain_point[1]:stain_point[3], stain_point[0]:stain_point[2]] = stain
    #img_ori = cv2.addWeighted(img_ori, 0.5, stain, 0.5, 0)

    img1 = img
    img2 = stain

    rows, cols, channels = img2.shape
    roi = img1[stain_point[0]:stain_point[2],stain_point[1]:stain_point[3]]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask_inv', mask_inv)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    print(stain_point)
    print(rows,cols)
    img_ori[stain_point[0]:rows+ stain_point[0], stain_point[1]:cols + stain_point[1]] = dst

    dst2 = cv2.resize(img_ori, (0, 0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_LINEAR)
    dst2 = cv2.flip(dst2, 1)
    return dst2



start = True

stain = cv2.imread("images/stain.png")
#stain = add_alpha_channel(stain)
stain = cv2.resize(stain, (0, 0), fx = 0.09, fy = 0.09, interpolation=cv2.INTER_LINEAR)

stain_point = [0, 0, 0, 0] #x,y,w,h

cap = cv2.VideoCapture(0)

print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
#    frame = add_alpha_channel(frame)
    if ret:
        frame = trans(frame)
        cv2.imshow('frame', frame)
        print(frame.shape)
        cv2.imshow("stain", stain)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()