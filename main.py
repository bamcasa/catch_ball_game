import cv2
import numpy as np
import random

def contact(contours):
    global stain_point,score
    for contour in contours:
        if contour['x'] <= stain_point[0] and contour['x'] + contour['w'] >= stain_point[0] and \
                contour['y'] <= stain_point[1] and contour['y'] + contour['h'] >= stain_point[1]:
            score += 1
            return True

def make_stain(img):
    global stain, stain_point
    height, width, channel = img.shape
    rand_Y = random.randint(50,height - 50)
    rand_X = random.randint(50,width - 50)
    stain_point[0] = rand_X
    stain_point[1] = rand_Y

def trans(img):
    global start, stain, stain_point
    if start:
        make_stain(img)
        start = False
    img_ori = img
    img_ori = cv2.flip(img_ori, 1)

    hsvim = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    MIN_RATIO, MAX_RATIO = 0.45, 1.1

    # MIN_AREA = 0
    # MIN_WIDTH, MIN_HEIGHT = 0, 0
    # MIN_RATIO, MAX_RATIO = 0, 10000000000000000


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
        cv2.drawContours(img_ori, d['contour'], -1, (255, 255, 0),5)
        cv2.rectangle(img_ori, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255), thickness=2)

    if contact(possible_contours):
        make_stain(img_ori)

    img_ori = cv2.circle(img_ori,(stain_point[0],stain_point[1]),50,(255,255,0), -1)
    cv2.putText(img_ori, f"scord : {score}", (1, 13*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    dst2 = cv2.resize(img_ori, (0, 0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_LINEAR)
    return dst2


score = 0
start = True

stain_point = [0, 0, 0, 0] #x,y,w,h

cap = cv2.VideoCapture(0)

print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = trans(frame)
        cv2.imshow('frame', frame)
        print(frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()