# Trương Văn Gia Lân - 19146204
# Trịnh Tuấn Vũ - 19146014 lớp chiều thứ 6
# Liên Hữu Lộc - 19146208
import cv2
import numpy as np
import time
TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
         'kcf' : cv2.legacy.TrackerKCF_create,
         'boosting' : cv2.legacy.TrackerBoosting_create,
         'mil': cv2.legacy.TrackerMIL_create,
         'tld': cv2.legacy.TrackerTLD_create,
         'medianflow': cv2.legacy.TrackerMedianFlow_create,
         'mosse':cv2.legacy.TrackerMOSSE_create}

trackers = cv2.legacy.MultiTracker_create()

# Green color in BGR
color = (0, 255, 0)
# Line thickness of 9 px
thickness = 2
# Ball count
count = 0

# v = cv2.VideoCapture('ball.mp4')
v = cv2.VideoCapture("ball.mp4")
cTime = 0
pTime = 0 
ret, frame = v.read()
pic_width = np.asarray(frame).shape[1]
pic_heigth = np.asarray(frame).shape[0]
pic_ratio = 0.2
pic_width_resize = pic_width - pic_width*pic_ratio
pic_height_resize = pic_heigth - pic_heigth*pic_ratio
frame = cv2.resize(frame,(int(pic_width_resize),int(pic_height_resize)),interpolation = cv2.INTER_AREA)


def hsv(frame):
    # frame = cv2.GaussianBlur(frame, (7,7), 0)
    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # yellow filter with inRange function
    low_H = 9
    hight_H = 21
    low_S = 110
    hight_S = 245
    low_V = 125
    hight_V = 255
    mask = cv2.inRange(hsv,np.array([low_H, low_S, low_V]),np.array([hight_H,hight_S,hight_V]))
    
    # Morphological Transformations,Opening and Closing
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def ball_detect(frame):
    # find contours
    contours, hierachy = cv2.findContours(frame,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for i, c in enumerate(contours):
        Area = cv2.contourArea(c)
        if Area > 20 and Area < 10000:
            x,y,w,h = cv2.boundingRect(c)
            balls.append([x,y,w,h])
    return balls
    



while True:
    ret, frame = v.read()
    frame = cv2.resize(frame,(int(pic_width_resize),int(pic_height_resize)),interpolation = cv2.INTER_AREA)
    h,w,_ = frame.shape

    # print(w)
    hsv_1 = hsv(frame)
    balls = ball_detect(hsv_1)
    trackers = cv2.legacy.MultiTracker_create()
    cv2.line(frame, (w-100,0), (w-100,h), color, thickness) ##  color  = gr, thickness = 2

    for i in range(len(balls)):
        tracker_i = TrDict['csrt']()
        trackers.add(tracker_i,frame,tuple(balls[i]))
    if not ret:
        break
    (success,boxes) = trackers.update(frame)
    # print((boxes))

    for i,box in enumerate(boxes):
        (x,y,w,h) = [int(a) for a in box]
        # print((w))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,256,0),2)
        center = center_handle(x,y,w,h)
        cv2.circle(frame,center,3,(0,0,255),-1)
        point = []
        point.append(center)
        # print((center))

        for (x,y) in point:
            if x < (600) and x > (590):
                count+=1
            point.remove((x,y))
            cv2.putText(frame,str(count+1),(x+10,y-3),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
        

    # SHOW FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    
    # SHOW FRAME 
    cv2.imshow('hsv',hsv_1)
    cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(10)
    if key == 27:
        break
v.release()
cv2.destroyAllWindows()