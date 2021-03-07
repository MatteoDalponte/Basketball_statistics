import os
import sys
import json
import datetime
import numpy as np
import random
import cv2
import colorsys
import skimage.io
#from google.colab.patches import cv2_imshow
from imutils.video import FPS
import math


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class Tracker:
    def __init__(self):
        self.initBB = None
        
        self.tracker = cv2.TrackerCSRT_create() 
        
        #tracker.save('tracker_params.yaml')
        #carica il file di configurazione del tracker
        file_path = 'tracker_params.yaml'
        self.fp = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)  # Read file
        self.tracker.read(self.fp.getFirstTopLevelNode())  # Do not use: tracker.read(fp.root())
        
        #storico della distanza tra il pallone travato da tracker ed il pallone trvoato dal detector
        
        self.history_distance = []
        
        
        
    def track(self,frame,boxes,scores,f,frame_id):
        (H, W) = frame.shape[:2]
        #draw bounding box of ball
        if len(boxes)>0:
            coor = boxes[0] #nel caso della palla il detector ritorna solo 1 detection ogni volta
            p1 = (int(coor[0]), int(coor[1]))
            p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 8, 5)
        
        #frame = draw_bbox(frame, [], complete, show_label=False, tracking=True)


	#caso 1: tracker non inizializzato, Bounding Box di inizio = none... alla prima detection inizializzo il tracker
        if self.initBB is None:            
            
            self.tracker = cv2.TrackerCSRT_create()
            #tracker = cv2.TrackerKCF_create()
            
            self.tracker.read(self.fp.getFirstTopLevelNode())
            
            print ("waiting for the first detection to inizialize the tracker!")
            for i, bbox in enumerate(boxes):
                if scores[0]>0.40:              #inizializzo il tracker se la confidence della detection è sufficientemente alta
                    coor = np.array(bbox[:4], dtype=np.int32)
                    self.initBB = (coor[0], coor[1], coor[2], coor[3])

                    self.tracker.init(frame, self.initBB)
                    fps = FPS().start()
                
	# detection presente, faccio un update del tracker (passo il nuovo  frame) 
        if self.initBB is not None: # vuol dire che è già stata fatta una prima detection
            (success, tracked_box) = self.tracker.update(frame)

            if success:
                p1 = (int(tracked_box[0]), int(tracked_box[1]))
                p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
                cv2.rectangle(frame, p1, p2, (255,0,100), 6, 3)
                
                f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame_id, int(tracked_box[0]), int(tracked_box[1]), int(tracked_box[2]), int(tracked_box[3]) , 1))
               
            else: 
                self.initBB = None
                print("tracking ha perso il targhet! in attesa di reinizializzazione")
        
        
        #caso 2 detection e tracker sono presenti... calcolo la distanza euclidea tra posizione della detection ed il tracker
        if (self.initBB is not None and len(boxes)>0):
            p1_tracker = (int(tracked_box[0]), int(tracked_box[1]))
            p2_tracker = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
            center_tracker = (int((p1_tracker[0]+p2_tracker[0])/2),int((p1_tracker[1]+p2_tracker[1])/2))
            #print(center_tracker)
            #print (boxes)
            bbox = boxes[0]
            coor = np.array(bbox[:4], dtype=np.int32)
            p1_prediction = (int(coor[0]), int(coor[1]))
            p2_prediction = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
            center_prediction = (int((p1_prediction[0]+p2_prediction[0])/2),int((p1_prediction[1]+p2_prediction[1])/2))
            #print(center_prediction)
            
            distance = math.sqrt((center_tracker[0]-center_prediction[0])**2 + (center_tracker[1]-center_prediction[1])**2)
            print("d=" + str(distance)+ "    d*conficence: "+str(distance*scores[0]))
            self.history_distance.append(distance*scores[0])
            
            print(self.history_distance[-5:])
            
            
             # se per ben 2 frame la distanza tra tracking e detection è troppo elevata reinizializzo il tracking
            if self.history_distance[-1]>20 and self.history_distance[-2]>30:
                
                self.history_distance[-1] = 0
                self.history_distance[-2] = 0
                
                
                print("distanza tra tracking e detection è troppo elevata reinizializzo il tracking d=" + str(distance))
                self.tracker = cv2.TrackerCSRT_create()
                #tracker = cv2.TrackerKCF_create()
                
                self.tracker.read(self.fp.getFirstTopLevelNode())
            

                for i, bbox in enumerate(boxes):
                    coor = np.array(bbox[:4], dtype=np.int32)
                    self.initBB = (coor[0], coor[1], coor[2], coor[3])
                    f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame_id, int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]) , 1))
                    p1 = (int(coor[0]), int(coor[1]))
                    p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,100), 6, 3)

                    self.tracker.init(frame, self.initBB)
                    fps = FPS().start()
            


#funziona chiamata ad ogni frame per ottenere la posizone del pallon... chamata a get_gt passando il frameID ed il dizionario ottenuto con get_dict
def get_gt(image, frame_id, gt_dict):
    if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
        return [], [], [], []
        
    frame_info = gt_dict[frame_id]
    
    detections = []
    out_scores = []
    ids = []
    complete = []
    
    for i in range(len(frame_info)):
        coords = frame_info[i]['coords']
        
        x1,y1,w,h = coords

        detections.append([x1,y1,w,h])
        out_scores.append(frame_info[i]['conf'])
        ids.append(1)

        complete.append([x1,y1,w,h,frame_info[i]['conf'],1])

    return detections,out_scores,ids, complete


# dato il file MOTA ottieni il dizionario contenente tutte le detection:
def get_dict(filename):
    with open(filename) as f:    
        d = f.readlines()
        
    d = list(map(lambda x:x.strip(),d))
    
    last_frame = int(d[-1].split(',')[0])
    gt_dict = {x:[] for x in range(last_frame+1)}
    
    for i in range(len(d)):
        a = list(d[i].split(','))
        a = list(map(float,a))    
        
        coords = a[2:6]
        confidence = a[6]

        gt_dict[a[0]].append({'coords':coords,'conf':confidence})
        
    return gt_dict


def opencv_tracking(video_path, detection_path, out_txt_file):
    gt_dict = get_dict(detection_path)
    frame_id = 1
    #apri file txt in cui salvarela bounding box del tracker
    try:
        os.remove(out_txt_file)
    except :
        None
    f = open(out_txt_file, "a")
    

    # Input
    video = cv2.VideoCapture(video_path)

    # Output
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #definisco formato output video mp4
    out = cv2.VideoWriter('output-finale.mp4',fourcc, 30.0, (int(video.get(3)),int(video.get(4))),True) #definisco proprietà output video
   
    
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    ret = True
    success = False
    initBB = None  

    fps = None
    
    tracker = Tracker()
	#ad ogni frame....
    while ret:
        print("Frame: {}".format(frame_id))
        ret, frame = video.read()

        if not ret:
            continue

        boxes, scores, names = [], [], []
        boxes,scores,names, complete = get_gt(frame,frame_id,gt_dict)
        
        
        tracker.track(frame,boxes,scores,f,frame_id)
        

        out.write(frame)    

        frame_id+=1

    out.release()

############################################################
#  Main Tracking
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--det', required=True,
                        metavar="/path/to/balloon/dataset/",
                        help='Path to detections file')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the tracking on')
    parser.add_argument('--out_tracker', required=True,
                        metavar="path of output tracker file",
                        help='path of output tracker file')
    args = parser.parse_args()

    print("Video: ", args.video)
    print("Detections: ", args.det)
    print("out: ", args.out_tracker)

    opencv_tracking(args.video, args.det, args.out_tracker)
