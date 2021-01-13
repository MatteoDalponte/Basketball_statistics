import os
import cv2
import sys
from detection import *

txt_filename = "det_tracking/det_yolo.txt"

#--------------aperuta files-----------------
cap = cv2.VideoCapture("nba.mp4") # FILE VIDEO INPUT
#cap = cv2.VideoCapture("out1.avi")
if not cap.isOpened():
    print ("impossiblie aprire video")
    sys.exit()
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('detection.mp4',fourcc, fps, (1920, 1080),True)
try:
	os.remove(txt_filename)
except :
	None
f = open(txt_filename, "a")



nframe=1
i=0

while (cap.isOpened()): #analisi video
    ret, frame = cap.read()
    if ret==True:
        image = frame       
        #cont,x,y,im,w,h=detect(image, "yolov3.cfg", "yolov3.weights", "yolov3.txt") #passo al programma la funzione detect presente in detection.py
        cont,x,y,im,w,h,score=detect(image, "weights/yolov3_ball.cfg", "weights/yolov3_2000_round2.weights", "weights/obj.names")
        image_to_show = cv2.resize(im, (1920, 1080))
        cv2.imshow("image",image_to_show)
        cv2.waitKey(1)
        
        #scrivo dati detection in json e un file txt        
        if(cont==True): #è stata individuata la palla            
            f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(nframe, x, y, w, h, score))            
            nframe=nframe+1
            i=i+1
        else: #non è stata individuata la palla            
            #f.write('{}, -1, {}, {}, {}, {}, {}, -1, -1, -1\n'.format(nframe, x, y, w, h, score))            
            nframe=nframe+1
            
        out.write(image_to_show) #output video
        
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        break



print("Numero Frame: ",nframe, " N detection palla: ",i,"\n")

f.close()
cap.release()
out.release()

