# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:35:30 2019

@author: Administrator
"""
import cv2 as cv 
import imutils
i=0
Side_length=54
Outer_frame=[[10, 10], [85, 10], [160, 10],
            [10, 85], [85, 85], [160, 85],
            [10, 160], [85, 160], [160, 160]
             ]

tup=() 
i=1
cap = cv.VideoCapture(0)
while(cap.isOpened):
    ret,frame = cap.read()
    frame = imutils.resize(frame, width=550)
    frame = frame[100:350, 150:400]
    img=frame.copy()
    for x,y in (Outer_frame):
        cv.rectangle(img,(x,y),(x+Side_length,y+Side_length),(255,255,255),1)
    cv.imshow("img",img)
    cv.imshow("frame",frame)
    if cv.waitKey(100) & 0xFF == 32:
        cv.imwrite('huanyuan%d.jpg'%i,frame)
        i=i+1
        print ('ok')
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
          
cap.release()
cv.destroyAllWindows()        
    
    
