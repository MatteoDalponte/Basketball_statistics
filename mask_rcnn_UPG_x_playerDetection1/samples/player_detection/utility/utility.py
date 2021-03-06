import os
import sys
import json
import datetime
import numpy as np
import random
import cv2
import colorsys
from time import sleep
from tqdm import tqdm

#Draw mutiple array of bbox on image
def draw_bbox(image, bboxes, det_bboxes, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False, colours=[]):   
    NUM_CLASS = ["back", "basketball"]
    num_classes = 2
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)

        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    for i, bbox in enumerate(det_bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        color = int(bbox[6])

        if color != -1 and colours != []:
            bbox_color = colours[color%32,:]
        else:
            bbox_color = (0, 0, 255)

        bbox_thick = int(0.8 * (image_h + image_w) / 1000)

        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick

        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[0] + coor[2], coor[1] + coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "Det {}".format(NUM_CLASS[class_ind])

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=2)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

#Get the information of a single frame
def get_gt(frame_id, gt_dict):
    if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
        return [], [], [], []
        
    frame_info = gt_dict[frame_id]
    
    detections = []
    out_scores = []
    ids = []
    complete = []
    
    for i in range(len(frame_info)):
        coords = frame_info[i]['coords']
        detect_ids = frame_info[i]['ids']
        
        x1,y1,w,h = coords

        detections.append([x1,y1,w,h])
        out_scores.append(frame_info[i]['conf'])
        ids.append(1)

        complete.append([x1,y1,w,h,frame_info[i]['conf'],1,detect_ids])

    return detections,out_scores,ids,complete

#Get the dict rappresentation of a txt file
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

        gt_dict[int(a[0])].append({'coords':coords,'conf':confidence,'ids':a[1]})
        
    return gt_dict

#Save dict version of tracking to mot txt (only one track for frame)
def save_mot(dic, txt):
    f = open(txt, "w")

    i = 0
    while (i in dic): 
        track, score, _, _ = get_gt(i, dic)

        if track != []:
            f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(i, track[0][0], track[0][1], track[0][2], track[0][3], score[0]))

        i+=1

#Save a video with the bbox
def save_to_video(det_dict, input_path, out_path):
    print("\n------------- Saving video.... ---------------\n")
    # Input video
    video = cv2.VideoCapture(input_path)
    length_input = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(video.get(3)),int(video.get(4))))

    frame_id = 0
    ret = True

    colours = np.random.rand(32, 3) * 255 #used only for display

    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while ret:
            ret, frame = video.read()

            if not ret:
                continue

            #Get bbox for single frame
            boxes, scores, names = [], [], []
            boxes,scores,names, complete = get_gt(frame_id, det_dict)

            (H, W) = frame.shape[:2]

            #Draw the detections boxes 
            frame = draw_bbox(frame, [], complete, show_label=False, tracking=True, colours=colours)

            #Save video frame
            out.write(frame)    
            frame_id+=1

            #Fancy print
            pbar.update(1)
            sleep(0.1)
    
    out.release()