import cv2 as cv
import numpy as np


font=cv.FONT_HERSHEY_TRIPLEX
box_side_length=50

class Start_point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.M_x=x+50+10
        self.M_y=y+50+35
          

class Creat_image(object):
    def __init__(self,x,y):
        self.image=np.zeros([x,y,3],np.uint8)
        self.image[:,:,0]=np.ones([x,y])*0
        self.image[:,:,1]=np.ones([x,y])*0
        self.image[:,:,2]=np.ones([x,y])*0

def draw_boxs(x,y,color):
    for i in range(3):
        for j in range(3):
            cv.rectangle(image.image,(x+i*box_side_length,y+j*box_side_length),
            (x+(i+1)*box_side_length-5,y+(j+1)*box_side_length-5),color,cv.FILLED)    


color=[(0, 255, 0),(255, 105, 65),(0, 165, 255),(255, 255, 255),(0, 0, 255),(0, 255, 255)]
location_names=['L','F','R','B','U','D']

top=Start_point(350,100)
left=Start_point(200,250)
mid=Start_point(350,250)
bottom=Start_point(350,400)
right1=Start_point(500,250)
right2=Start_point(650,250)
image=Creat_image(600,1000)
locations={'L':(left.M_x,left.M_y),'F':(mid.M_x,mid.M_y),'R':(right1.M_x,right1.M_y),
            'B':(right2.M_x,right2.M_y),'U':(top.M_x,top.M_y),'D':(bottom.M_x,bottom.M_y)}
string='Xiao Mo\'s magic cube'
while True:
    
    cv.putText(image.image,string,(250,50),font,1.2,(159,255,84),1)
    draw_boxs(top.x,top.y,color[0])
    draw_boxs(left.x,left.y,color[1])
    draw_boxs(mid.x,mid.y,color[2])
    draw_boxs(bottom.x,bottom.y,color[3])
    draw_boxs(right1.x,right1.y,color[4])
    draw_boxs(right2.x,right2.y,color[5])
    for i , location in zip([_ for _ in range (6)],locations.values()):
        cv.putText(image.image,location_names[i],location,font,1.2,(0,0,0),1)
    cv.imshow('frame',image.image)
    cv.waitKey(5)
    if cv.waitKey(5)&0xff==ord('q'):
        break
cv.destroyAllWindows()