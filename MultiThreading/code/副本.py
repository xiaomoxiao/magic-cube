# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:12:54 2019

@author: Administrator
"""

import cv2 as cv
import numpy as np
import imutils
import json



Side_length=54
Outer_frame=[[10, 10], [85, 10], [160, 10],
            [10, 85], [85, 85], [160, 85],
            [10, 160], [85, 160], [160, 160]
             ]
###绿色

rands=[[],[],[],[],[],[]]
number_of_data=[[],[],[],[],[],[]]
for file_number in range(6):
    file_name='./test{0}.json'.format(file_number)
    with open(file_name,'r') as f:
        rands[file_number]=json.load(f)
        number_of_data[file_number]=len(rands[file_number])
    f.close()

lable_lists=[[0] for _ in range ((number_of_data[0]))]
lable_lists.extend([[4] for _ in range ((number_of_data[1]))])
lable_lists.extend([[1] for _ in range ((number_of_data[2]))])
lable_lists.extend([[5] for _ in range ((number_of_data[3]))])
lable_lists.extend([[2] for _ in range ((number_of_data[4]))])
lable_lists.extend([[3] for _ in range ((number_of_data[5]))])



'''
#绿色
rand1 = np.array([[83, 248, 136, 4, 2, 1], [79, 246, 115, 3, 4, 2], [79, 246, 115, 3, 4, 2], [83, 248, 136, 4, 2, 1], [79, 246, 115, 3, 4, 2], [79, 246, 115, 3, 4, 2], [83, 248, 136, 4, 2, 1], [79, 246, 115, 3, 3, 1], [79, 246, 115, 3, 4, 2]])
#蓝色
rand2 = np.array([[107, 251, 186, 1, 1, 2], [109, 254, 174, 0, 0, 1], [109, 254, 174, 0, 0, 1], [106, 250, 195, 2, 3, 4], [109, 254, 174, 0, 0, 1], [109, 254, 174, 0, 0, 1], [106, 250, 195, 2, 3, 4], [108, 252, 179, 0, 0, 1], [106, 251, 188, 1, 2, 3]])
#橙色
rand3 = np.array([[9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201], [9, 215, 206, 138, 206, 201]])
#白色
rand4=([[87, 16, 171, 88, 52, 160], [78, 12, 165, 110, 54, 156], [66, 8, 157, 142, 58, 152], [87, 16, 171, 88, 52, 160], [87, 16, 171, 88, 52, 160], [87, 16, 171, 88, 52, 160], [87, 16, 171, 88, 52, 160], [87, 16, 171, 88, 52, 160], [87, 16, 171, 88, 52, 160]])
#红色
rand5=([[3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147], [3, 198, 151, 124, 151, 147]])
#黄色
rand6=([[51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37], [51, 94, 185, 137, 47, 37]])


# lable = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
                #   [1],[1],[1],[1],[1],[1],[1],[1],[1],
                #   [2],[2],[2],[2],[2],[2],[2],[2],[2],
                #   [3],[3],[3],[3],[3],[3],[3],[3],[3],
                #   [4],[4],[4],[4],[4],[4],[4],[4],[4],
                #   [5],[5],[5],[5],[5],[5],[5],[5],[5]
                #   ])
'''

lable=np.array(lable_lists)

data = np.vstack((rands[0],rands[1],rands[2],rands[3],rands[4],rands[5]))
data = np.array(data,dtype='float32')

svm = cv.ml.SVM_create() #创建SVM model
#属性设置
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setC(1)

result = svm.train(data,cv.ml.ROW_SAMPLE,lable)
svm.save("svmtest.mat")

def get_color(x,y):
    hsv=[]
    list1=[]
    h=s=v=0
    img=frame[y+20:y+40,x+20:x+40]
    hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
    width = hsv.shape[0]
    height= hsv.shape[1]
    for index1 in range (width):
        for index2 in range (height):
            h=h+ hsv[index1,index2,0]
            s=s+ hsv[index1,index2,1]
            v=v+ hsv[index1,index2,2]
    aveh=h//(width*height)
    aves=s//(width*height)
    avev=v//(width*height)
    list1.append(aveh)
    list1.append(aves)
    list1.append(avev)
    return (list1)

def get_averageBGR(x,y):
    
    img=frame[y+10:y+50,x+10:x+50]
    img = cv.cvtColor(img,cv.COLOR_HSV2RGB)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    list1=[]
    per_image_Bmean.append(np.mean(img[:,:,0]))
    per_image_Gmean.append(np.mean(img[:,:,1]))
    per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    list1.append(R_mean)
    list1.append(G_mean)
    list1.append(B_mean)
    
    return list1


#pt_data = np.vstack([[100,120,60]])
#        
#pt_data = np.array(pt_data,dtype='float32')
#print(pt_data)
#(par1,par2) = svm.predict(pt_data)
#print(par1,par2)
#
#pt_data = np.vstack([[100,120,60]])
#        
#pt_data = np.array(pt_data,dtype='float32')
#print(pt_data)
#(par1,par2) = svm.predict(pt_data)
#print(par1,par2)
#cap= cv.VideoCapture(1)
frame=cv.imread("svmwo1.jpg")
cv.imshow("frame1",frame)
#while (1):
#    ret,frame=cap.read()
#    frame = imutils.resize(frame, width=550)
#    frame = frame[50:300, 150:400] 

for x,y in (Outer_frame):
    Hsv=get_color(x,y)
    Rgb=get_averageBGR(x,y)
    Rgb = list(map(int,Rgb)) 
    listall=Hsv+Rgb
    pt_data = np.vstack([listall])
    pt_data = np.array(pt_data,dtype='float32')
    (par1,par2) = svm.predict(pt_data)
    print (par2)
    if (par2==0):
        #####绿色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(0, 255, 0),-1)
    if (par2==1):
        #蓝色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(255, 105, 65),-1)
    if (par2==2):
        #橙色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(0, 165, 255),-1)  
    if (par2==3):
        #白色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(255, 255, 255),-1) 
    if (par2==4):
        #红色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(0, 0, 255),-1)
    if (par2==5):
        #黄色
        cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(0, 255, 255),-1)


#        else:
#            cv.rectangle(frame,(x,y),(x+Side_length,y+Side_length),(255,255,255),-1)       
cv.imshow("frame",frame)
#    if cv.waitKey(100) & 0xFF == ord('q'):
#        break
cv.waitKey(0)
cv.destroyAllWindows()
            
        
        
    