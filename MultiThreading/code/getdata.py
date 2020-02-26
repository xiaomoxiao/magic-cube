#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   getdata.py
@Time    :   2020/02/14 18:16:06
@Author  :   Gui XiaoMo 
@Version :   1.0
@Contact :   2572225959@qq.com
@License :   (C)Copyright 2020-2021, QINGDAO-QUST
@Desc    :   None
'''

# here put the import lib


import os 
import cv2 as cv 
import numpy as np
import time
import json
import threading
from queue import Queue
import sys


picture_path='C:/Users/Administrator/Desktop/1/'

picture_number=0  #第几个图片
num=0  #成功了多少张图片

#魔方的颜色
greenLower = (46, 133, 46)
greenUpper = (85, 255, 255)

redLower = (150, 100, 6)
redUpper = (185, 255, 255)

yellowLower = (21, 84, 46)
yellowUpper = (64, 255, 255)

orangeLower = (2, 150, 100)
orangeUpper = (15, 255, 255)

whiteLower = (0, 0, 146)  # gray
whiteUpper = (180, 78, 255)

blueLower = (88, 143, 46)
blueUpper = (120, 255, 255)


Side_length=54
Outer_frame=[[10, 10], [85, 10], [160, 10],
            [10, 85], [85, 85], [160, 85],
            [10, 160], [85, 160], [160, 160]
             ]

listnet=[]
listall=[]
listhsv=[]
listrgb=[]




class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


#获取图片的路径（返回图片路径）
def read_picture(i):
    path=picture_path+'huanyuan{0}.jpg'.format(i)
    print(path)

    return(path)


def indextocolor(index):
    color=()
    if (index==0):
        color=(0, 0, 255)
    if (index==1):
        color=(255, 0, 0)
    if (index==2):
        color=(0, 255, 255)
    if (index==3):
        color=(0, 165, 255)      
    if (index==4):
        color=(0, 255, 0)        
    if (index==5):
        color=(255, 255, 255)
    return (color)



    
def draw_rectangle(image,color,i):
    x=Outer_frame[i][0]
    y=Outer_frame[i][1]
    x1=Outer_frame[i][0]+Side_length
    y1=Outer_frame[i][1]+Side_length
    cv.rectangle(image,(x,y),(x1,y1),color,-1)
        
    
    
    

def get_averageBGR(image,x,y):
    img = cv.cvtColor(image,cv.COLOR_HSV2RGB)
    img=img[x+20:x+45,y+20:y+45]
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
    
    return (list1)



def get_averageHSV(img,x,y):
    hsv=[]
    list1=[]
    h=s=v=0
    image1=img[x+20:x+45,y+20:y+45]
    hsv= cv.cvtColor(image1,cv.COLOR_BGR2HSV)
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




def average(img):
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    
    
    
    image_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    #直方图均衡化
    image_yuv[:,:,0] = cv.equalizeHist(image_yuv[:,:,0])
    #显示效果
    output = cv.cvtColor(image_yuv,cv.COLOR_YUV2BGR)
    cv.imshow('HistEqualize',output)
    return (output)




#    img=cv.cvtColor(img,cv.COLOR_BGR2HSV)
#    (b, g, r) = cv.split(img)
#    bH = cv.equalizeHist(b)
#    gH = cv.equalizeHist(g)
#    rH = cv.equalizeHist(r)
#    # 合并每一个通道
#    result = cv.merge((bH, gH, rH))
#    cv.imshow("直方图均衡化", result)


def balance(img_input):

#    完美反射白平衡
#    STEP 1：计算每个像素的R\G\B之和
#    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
#    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
#    STEP 4：对每个点将像素量化到[0,255]之间
#    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
#    :param img: cv2.imread读取的图片数据
#    :return: 返回的白平衡结果图片数据

    img = img_input.copy()
    b, g, r = cv.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r
 
    return (img)


def gaussi_blur(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    #cv.imshow("gaussian",blur)
    return (blur)    




def k_means(img):
    
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    # convert to np.float32
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    #cv.imshow("k_means",res2)
    return (res2)

'''
image= cv.imread("huanyuan32.jpg")

cv.imshow("image",image)

img1=gaussi_blur(image)

img2=k_means(img1)
cv.imwrite("svmwo1.jpg",img2)

img3=balance(img2)
cv.imshow("balance",img3)
img4=average(img3)
#cv.imwrite("svmwo5.jpg",img4)

'''


def main(src):
    img1=gaussi_blur(src)
    img2=k_means(img1)
    for x,y in (Outer_frame):
        listhsv=get_averageHSV(img2,x,y)
        listrgb=get_averageBGR(img2,x,y)
        listrgb = list(map(int,listrgb)) 
        listnet=listhsv+listrgb
        listall.append(listnet)
    #print(listall)    


#########################多线程尝试#############################################



cube_list_hsv=[[] for _ in range (6)]
cube_list_bgr=[[] for _ in range (6)]
cube_list_all=[[] for _ in range (6)]
cube_list_net=[[] for _ in range (6)]


dict_data={"1":cube_list_all[0],'2':cube_list_all[1],'3':cube_list_all[2],
            '4':cube_list_all[3],'5':cube_list_all[4],'6':cube_list_all[5]
}

####多线程分别进行魔方6个面的识别
def job1():
    for i in range (1,29):
        path1 = read_picture(i)
        print (path1,end='\n')
        cube_list_hsv[0]=[]
        cube_list_bgr[0]=[]
        cube_list_net[0]=[]
        src1=cv.imread(path1)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[0]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[0]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[0]=list(map(int,cube_list_bgr[0]))
            cube_list_net[0]=cube_list_hsv[0]+cube_list_bgr[0]
            cube_list_all[0].append(cube_list_net[0])
    #q.put(cube_list_all[0])


def job2():
    for i in range (29,63):
        path2 = read_picture(i)
        # print (path1,end='\n')
        cube_list_hsv[1]=[]
        cube_list_bgr[1]=[]
        cube_list_net[1]=[]
        src1=cv.imread(path2)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[1]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[1]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[1]=list(map(int,cube_list_bgr[1]))
            cube_list_net[1]=cube_list_hsv[1]+cube_list_bgr[1]
            cube_list_all[1].append(cube_list_net[1])
    #q.put(cube_list_all[0])




def job3():
    for i1 in range (63,91):
        path1 = read_picture(i1)
        print (path1,end='\n')
        cube_list_hsv[2]=[]
        cube_list_bgr[2]=[]
        cube_list_net[2]=[]
        src1=cv.imread(path1)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[2]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[2]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[2]=list(map(int,cube_list_bgr[2]))
            cube_list_net[2]=cube_list_hsv[2]+cube_list_bgr[2]
            cube_list_all[2].append(cube_list_net[2])
    #q.put(cube_list_all[0])



def job4():
    for i1 in range (91,166):
        path1 = read_picture(i1)
        print (path1,end='\n')
        cube_list_hsv[3]=[]
        cube_list_bgr[3]=[]
        cube_list_net[3]=[]
        src1=cv.imread(path1)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[3]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[3]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[3]=list(map(int,cube_list_bgr[3]))
            cube_list_net[3]=cube_list_hsv[3]+cube_list_bgr[3]
            cube_list_all[3].append(cube_list_net[3])
    #q.put(cube_list_all[0])


def job5():
    for i1 in range (205,304):
        path1 = read_picture(i1)
        print (path1,end='\n')
        cube_list_hsv[4]=[]
        cube_list_bgr[4]=[]
        cube_list_net[4]=[]
        src1=cv.imread(path1)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[4]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[4]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[4]=list(map(int,cube_list_bgr[4]))
            cube_list_net[4]=cube_list_hsv[4]+cube_list_bgr[4]
            cube_list_all[4].append(cube_list_net[4])
    #q.put(cube_list_all[0])



def job6():
    for i1 in range (304,416):
        path1 = read_picture(i1)
        print (path1,end='\n')
        cube_list_hsv[5]=[]
        cube_list_bgr[5]=[]
        cube_list_net[5]=[]
        src1=cv.imread(path1)
        # if not src1:
        #     print('error reading picture')
        #     sys.exit()
        cube1_img1=gaussi_blur(src1)
        cube1_img2=k_means(cube1_img1)
        for x,y in (Outer_frame):
            cube_list_hsv[5]=get_averageHSV(cube1_img2,x,y)
            cube_list_bgr[5]=get_averageBGR(cube1_img2,x,y)
            cube_list_bgr[5]=list(map(int,cube_list_bgr[5]))
            cube_list_net[5]=cube_list_hsv[5]+cube_list_bgr[5]
            cube_list_all[5].append(cube_list_net[5])
    #q.put(cube_list_all[0])





'''
q=Queue()
threads=[]

t1 = threading.Thread(target=job1,name=('t1',))
t2 = threading.Thread(target=job2,name=('t2',))
t3 = threading.Thread(target=job3,name=('t3',))
t4 = threading.Thread(target=job4,name=('t4',))
t5 = threading.Thread(target=job5,name=('t5',))
t6 = threading.Thread(target=job6,name=('t6',))
t1.start()
threads.append(t1)
t2.start()
threads.append(t2)
t3.start()
threads.append(t3)
t4.start()
threads.append(t4)
t5.start()
threads.append(t5)
t6.start()
threads.append(t6)
for thread in threads:
    thread.join()
print('all pictures are taken\n')
'''




#every_data_contain_number

#for key in dict_data:


number_of_dict=len(dict_data)

#声明6个，用来作为文本存储，json不支持numpy 的int32  我用本办法转换

store_data=[[] for _ in range (number_of_dict)]

#把这几个数组百变成字典中列表的格式

for circule_num,value in  zip([x for x in range(0,6)],dict_data.values()):
    store_data[circule_num] = [[0,0,0,0,0,0] for i in range (len(value))]
    for first in range(len(value)):
        for two in range(len(value[first])):
            store_data[circule_num][first][two]=int(value[first][two])


for json_number in range (6):
    file_name="data{0}.json".format(json_number)
    with open(file_name,"w") as f:
        json.dump(store_data[json_number],f)
    f.close()














'''
for i in range(1,29):

    path=read_picture(i)
    print (path)
    listhsv.clear()#清空hsv的tup
    listrgb.clear()#清空rgb的tup
    listnet.clear()#清空节点的tup
    src = cv.imread(path)
    while (src is None):
        src = cv.imread(path)
        if not src:
            print('error reading picture')
            sys.exit()
    main(src)
print(listall)
print ('个数是')

list_num=len(listall)
store = [[0,0,0,0,0,0] for i in range (list_num)]
for list_1 in range(len(listall)):
    for list_2 in range(len(listall[list_1])):
        store[list_1][list_2]=int(listall[list_1][list_2])

'''











'''
filename='test.json'


with open(filename,'w') as f:
    json.dump(store,f)
f.close()
'''



'''
with open('test(副本).txt','w') as f1:
    for temp in listall:
        print(type(temp[0]))
        data='{},{},{},{},{},{}\n'.format(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5])
        f1.write(data)
        
f1.close()
'''



