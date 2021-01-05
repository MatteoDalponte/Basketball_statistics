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

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library



def get_gt(image, frame_id, gt_dict):
    if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
        return [], [], [], [], []
        
    frame_info = gt_dict[frame_id]
    
    detections = []
    out_scores = []
    ids = []
    complete = []
    team = []
    
    for i in range(len(frame_info)):
        coords = frame_info[i]['coords']
        
        x1,y1,w,h = coords

        detections.append([x1,y1,w,h])
        out_scores.append(frame_info[i]['conf'])
        ids.append(1)
        team.append(frame_info[i]['team'])

        complete.append([x1,y1,w,h,frame_info[i]['conf'],1])

    return detections,out_scores,ids, complete, team


# conversion of the file txt to dictionary NOTE: modified to return the team number
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
        try:
            team = a[10]        #get the team number
        except:
            team =[]

        gt_dict[a[0]].append({'coords':coords,'conf':confidence,'team':team})
        
    return gt_dict


def draw_players(frame, boxes_team,team_numbers):
    for i, box in enumerate(boxes_team):
        coor = boxes_team[i]
        p1 = (int(coor[0]), int(coor[1]))
        p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
        #print(team_numbers)
        
        number = team_numbers[i]
        if number == 0:
            cv2.rectangle(frame, p1, p2, (100,0,0), 6, 3)
        elif number == 1:
            cv2.rectangle(frame, p1, p2, (0,0,0), 6, 3)
        elif number == 2:
            cv2.rectangle(frame, p1, p2, (255,255,255), 6, 3)
        else:
            cv2.rectangle(frame, p1, p2, (100,100,100), 6, 3)
    return frame

def draw_rect(frame, coor,colour):
   
        p1 = (int(coor[0]), int(coor[1]))
        p2 = (int(coor[0] + coor[2]), int(coor[1] + coor[3]))
        cv2.rectangle(frame, p1, p2, colour, 7, 3)
        
        return frame
        

def distance_boxes(box1,box2):
    p1_box1 = (int(box1[0]), int(box1[1]))
    p2_box1 = (int(box1[0] + box1[2]), int(box1[1] + box1[3]))
    center_box1 = (int((p1_box1[0]+p2_box1[0])/2),int((p1_box1[1]+p2_box1[1])/2))
    
    p1_box2 = (int(box2[0]), int(box2[1]))
    p2_box2 = (int(box2[0] + box2[2]), int(box2[1] + box2[3]))
    center_box2 = (int((p1_box2[0]+p2_box2[0])/2),int((p1_box2[1]+p2_box2[1])/2))
            
            
    distance = math.sqrt((center_box1[0]-center_box2[0])**2 + (center_box1[1]-center_box2[1])**2)
    return distance


line_points = []

def draw_blue_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #center = (100,100)
        #radius = calc_distance((x, y), center)     
        cv2.circle(i, (x,y), 2, (255, 0, 0), 2)
        line_points.append((x*3,y*3))



def stats(video_path, ball_tracking_path, team_detection_path, out_txt_file):
    
    ball_dict = get_dict(ball_tracking_path)
    team_dict = get_dict(team_detection_path)
    
    frame_id = 1
    #output file for statistics
    try:
        os.remove(out_txt_file)
    except :
        None
    f = open(out_txt_file, "a")

    possesso_palla = np.array([0, 0, 0])
    ball_cumulative_position = np.array([0, 0])
    
    # Input video
    video = cv2.VideoCapture(video_path)

    # Output video
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #definisco formato output video mp4
    out = cv2.VideoWriter('output-finale.mp4',fourcc, 30.0, (int(video.get(3)),int(video.get(4))),True) #definisco proprietà output video
   
    
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
     
    
    #acquisisci linea metà campo
    cv2.namedWindow("select 2 point of the center line")
    cv2.setMouseCallback("select 2 point of the center line", draw_blue_circle) # param = None
    ret, img = video.read()
    (H, W) = img.shape[:2]
    global i
    i = cv2.resize(img, (int(W/3), int(H/3)))
    cv2.putText(i,"select 2 extreme points of the middle line and press Q",(5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    while True:
        # both windows are displaying the same img
        cv2.imshow("select 2 point of the center line", i)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    
    print(line_points)

    ret = True
    

    while ret:
        #print("Frame: {}".format(frame_id))
        ret, frame = video.read()

        if not ret:
            continue
        #image = frame.copy()
        image = frame

        #estrazione boxes teams e palla dai rispettivi dataset
        boxes_team, scores_team, names_team, team_numbers = [], [], [], []
        boxes_team,scores_team,names_team, complete_team, team_numbers = get_gt(frame,frame_id,team_dict)                    
        
        
        boxes_ball, scores_ball, names_ball, not_used= [], [], [], []
        boxes_ball,scores_ball,names_ball, complete_ball, not_used = get_gt(frame,frame_id,ball_dict)
        
        #
        image = draw_players(image,boxes_team,team_numbers)
        if len(boxes_ball)>0:
            coor = boxes_ball[0] #nel caso della palla il detector ritorna solo 1 detection ogni volta
            draw_rect(image, coor,(10,255,255))
            
        
        # possesso palla teams
        if(len(boxes_ball) >0):
            ball_players_distance = []        
            for box in boxes_team:
                ball_players_distance.append(distance_boxes(box,boxes_ball[0]))
            
            player_index = np.argmin(ball_players_distance)
                    
            #cv2.line(image,(int(boxes_team[player_index][0]),int(boxes_team[player_index][1])),(int(boxes_ball[0][0]),int(boxes_ball[0][1])),(255,0,0),5)
            if(ball_players_distance[player_index]<200):
                team_number = int(team_numbers[player_index])
                possesso_palla[team_number] = possesso_palla[team_number] + 1
                
                image = draw_rect(image, boxes_team[player_index],(0,0,255))
                if (team_numbers[player_index]) == 0:
                    txt = "arbitro"
                if (team_numbers[player_index]) == 1:
                    txt = "squadra A"
                if (team_numbers[player_index]) == 2:
                    txt = "squadra B"
                cv2.putText(
                    image, #numpy array on which text is written
                    txt, #text
                    (int(boxes_team[player_index][0]),int(boxes_team[player_index][1]-5)), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    1, #font size
                    (0, 0, 0, 255), #font color
                    3) #font stroke
                
            #posizione palla metà campo DX o SX
            line_points_arr = np.asarray(line_points)
            p1 =[coor[0],coor[1]]
            distance_ball_center = (np.cross(line_points_arr[0]-p1, p1-line_points_arr[1]))/np.linalg.norm(line_points_arr[0]-p1)
            if(distance_ball_center<0):
                ball_cumulative_position[0] = ball_cumulative_position[0] +1
            else:
                ball_cumulative_position[1] = ball_cumulative_position[1] +1
                
        #show image       
        
        image_to_show = cv2.resize(image, (1920, 1080))
        cv2.imshow("image",image_to_show)
        cv2.waitKey(1)
        

        #out.write(image)    

        frame_id+=1
        
    f.write("-----------------possesso palla:------------------- \n \n")
    txt1 = "arbitro (frame/total_frame): "+str(possesso_palla[0])+" / "+ str(frame_id)+"      ->   "+str(int(possesso_palla[0]/frame_id*100))+"% \n"
    f.write(txt1)
    txt2 = "team A (frame/total_frame): "+str(possesso_palla[1])+" / "+ str(frame_id)+"       ->   "+str(int(possesso_palla[1]/frame_id*100)) +"% \n"
    f.write(txt2)
    txt3 = "team B (frame/total_frame): "+str(possesso_palla[2])+" / "+ str(frame_id)+"       ->   "+str(int(possesso_palla[2]/frame_id*100)) +"% \n"
    f.write(txt3)
    
    f.write("\n \n-----------------ball cumulative position:------------------- \n \n")
    txt4 = "SX (frame/total_frame): "+str(ball_cumulative_position[0])+" / "+ str(frame_id)+"       ->   "+str(int(ball_cumulative_position[0]/frame_id*100)) +"% \n"
    f.write(txt4)
    txt5 = "DX (frame/total_frame): "+str(ball_cumulative_position[1])+" / "+ str(frame_id)+"       ->   "+str(int(ball_cumulative_position[1]/frame_id*100)) +"% \n"
    f.write(txt5)
    
    out.release()

############################################################
#  Main statistics
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--det_teams', required=True,
                        metavar="/path/to/balloon/dataset/",
                        help='Path to detections file of the team')
    parser.add_argument('--ball_track', required=True,
                        metavar="/path/to/balloon/dataset/",
                        help='Path to track of the ball')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the tracking on')
    parser.add_argument('--out_stats', required=True,
                        metavar="path of output tracker file",
                        help='path of output tracker file')
    args = parser.parse_args()

    print("Video: ", args.video)
    print("Detections: ", args.det_teams)
    print("Ball_track: ", args.ball_track)
    print("out: ", args.out_stats)

    stats(args.video, args.ball_track, args.det_teams, args.out_stats)
