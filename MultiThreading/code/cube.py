import cv2 as cv 
import numpy as np

file_path='./testX/'
svm2 = cv.ml.SVM_load("svmtest.mat")

Lower_red = np.array([0, 43, 46])#要识别颜色的下限
Upper_red = np.array([10, 255, 255])#要识别的颜色的上限

Lower_orange = np.array([11, 43, 46])#要识别颜色的下限

Upper_orange = np.array([25, 255, 255])#要识别的颜色的上限

flag=1
i=1

Side_length=54
Outer_frame=[[10, 10], [85, 10], [160, 10],
            [10, 85], [85, 85], [160, 85],
            [10, 160], [85, 160], [160, 160]
             ]

def read_picture(i):
    path=file_path+'huanyuan{0}.jpg'.format(i)
    print(path)
    return(path)

def gaussi_blur(img):
    blur = cv.GaussianBlur(img,(9,9),0)
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

def get_color(image,x,y):
    hsv=[]
    list1=[]
    h=s=v=0
    img=image[y+15:y+45,x+15:x+45]
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

def get_averageBGR(image,x,y):
    
    img=image[y+15:y+45,x+15:x+45]
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

    return (list1)


while True:
    path = read_picture(i) 
    
    if (flag==1):
        img=cv.imread(path)
        src=img
        print('hello')
        cv.imshow('frame1',img)
        for x,y in (Outer_frame):
            img[y:y+Side_length,x:x+Side_length]=gaussi_blur(img[y:y+Side_length,x:x+Side_length])
            img[y:y+Side_length,x:x+Side_length]=k_means(img[y:y+Side_length,x:x+Side_length])
            Hsv=get_color(img,x,y)
            Rgb=get_averageBGR(img,x,y)
            Rgb = list(map(int,Rgb)) 
            listall=Hsv+Rgb
            pt_data = np.vstack([listall])
            pt_data = np.array(pt_data,dtype='float32')
            (par1,par2) = svm2.predict(pt_data)
            print (par2)

            if (par2==0):
                #####绿色
                cv.rectangle(img,(x,y),(x+Side_length,y+Side_length),(0, 255, 0),-1)
            if (par2==1):
                #蓝色
                cv.rectangle(src,(x,y),(x+Side_length,y+Side_length),(255, 105, 65),-1)
            if (par2==2):
                #橙色
                cv.rectangle(src,(x,y),(x+Side_length,y+Side_length),(0, 165, 255),-1)  
            if (par2==3):
                #白色
                cv.rectangle(src,(x,y),(x+Side_length,y+Side_length),(255, 255, 255),-1) 
            if (par2==4):
                #红色
                cv.rectangle(src,(x,y),(x+Side_length,y+Side_length),(0, 0, 255),-1)
            if (par2==5):
                #黄色
                cv.rectangle(src,(x,y),(x+Side_length,y+Side_length),(0, 255, 255),-1)

            #（4，2）
            if ((par2==2) or(par2==4)):
                the_img=img[y:y+Side_length,x:x+Side_length]
                hsv_the_img=cv.cvtColor(the_img,cv.COLOR_BGR2HSV)
 



        flag=flag-1
    cv.imshow("frame",src)
    if cv.waitKey(10) & 0xFF == ord('c'):
        cv.destroyAllWindows()
        i=i+1
        flag=1
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cv.destroyAllWindows() 
    
        
            