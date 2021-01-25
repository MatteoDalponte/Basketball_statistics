import os
import sys
import numpy as np
import cv2
import math     
        
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

def circle_player(image, box1, radius):
    p1_box1 = (int(box1[0]), int(box1[1]))
    p2_box1 = (int(box1[0] + box1[2]), int(box1[1] + box1[3]))
    center_box1 = (int((p1_box1[0]+p2_box1[0])/2),int((p1_box1[1]+p2_box1[1])/2))
    cv2.circle(image, tuple(center_box1), radius, (0, 0, 200), 2)

def ball_position(line_a, line_b, point):
    if line_a[1] < line_b[1]:
        line_a, line_b = line_b, line_a
    return np.sign((line_b[0] - line_a[0])*(point[1] - line_a[1]) - (line_b[1] - line_a[1])*(point[0]- line_a[0]))
    