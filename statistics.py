import os
import sys
import numpy as np
import cv2
import math

from utility.stat_utility import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

class Statistics:
    def __init__(self):
        self.line_points = []
        self.resize = 1

        #variables for statistics 1:
        self.possesso_palla = np.array([0, 0, 0])
        self.ball_cumulative_position = np.array([0, 0])
        self.last_valid_ball = []
        self.storia_possesso_palla = []
        self.filtered_team_number = []  #team number con possesso palla
        
        #self.history_players_near_ball = []
        
        #per statistica 4
        self.history_mean_dist_team1= []
        self.history_mean_dist_team2= []
        
        self.ballDX = False
        self.ballSX = False
        self.history_distance_ball_center = []
        
        #per statistica 5:
        self.pressione = np.array([0, 0])
        
    def initialize(self, img, resize):
        self.resize = resize

        cv2.namedWindow("selectpoint")
        cv2.setMouseCallback("selectpoint", self.draw_line) # param = None

        (H, W) = img.shape[:2]

        global i
        i = cv2.resize(img, (int(W/3), int(H/3)))
        cv2.putText(i, "Select 2 extreme points of the middle line and press Q", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        while True:
            # both windows are displaying the same img
            cv2.imshow("selectpoint", i)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        #print(self.line_points)
        
    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #center = (100,100)
            #radius = calc_distance((x, y), center)     
            cv2.circle(i, (x, y), 2, (255, 0, 0), 2)
            self.line_points.append((int((x*3)/self.resize), int((y*3)/self.resize)))

    def generate_file(self,f, frame_id):
        f.write("-----------------possesso palla:------------------- \n \n")
        txt1 = "arbitro (frame/total_frame): " + str(self.possesso_palla[0]) + " / " + str(frame_id) + "      ->   " + str(int(self.possesso_palla[0] / frame_id * 100)) + "% \n"
        f.write(txt1)
        txt2 = "team A (frame/total_frame): " + str(self.possesso_palla[1]) + " / " + str(frame_id) + "       ->   " + str(int(self.possesso_palla[1] / frame_id * 100)) + "% \n"
        f.write(txt2)
        txt3 = "team B (frame/total_frame): " + str(self.possesso_palla[2]) + " / " + str(frame_id) + "       ->   " + str(int(self.possesso_palla[2] / frame_id * 100)) + "% \n"
        f.write(txt3)
        
        f.write("\n \n-----------------ball cumulative position:------------------- \n \n")
        txt4 = "SX (frame/total_frame): " + str(self.ball_cumulative_position[0]) + " / " + str(frame_id) + "       ->   " + str(int(self.ball_cumulative_position[0] / frame_id * 100)) + "% \n"
        f.write(txt4)
        txt5 = "DX (frame/total_frame): " + str(self.ball_cumulative_position[1]) + " / " + str(frame_id) + "       ->   " + str(int(self.ball_cumulative_position[1] / frame_id * 100)) + "% \n"
        f.write(txt5)
        
        f.write("\n \n-----------------pressione difesa:------------------- \n \n")
        txt6 = "affollamenti a SX: " + str(self.pressione[0]) + " / " + str(frame_id) + "       ->   " + str(int(self.pressione[1] / np.sum(self.pressione) * 100)) + "% \n"
        f.write(txt6)
        txt7 = "affollamenti a SX:: " + str(self.pressione[1]) + " / " + str(frame_id) + "       ->   " + str(int(self.pressione[0] / np.sum(self.pressione) * 100)) + "% \n"
        f.write(txt7)
        
        return f
        
    # Generazione statistica 1
    def stat1(self,image,boxes_ball,boxes_team,team_numbers,fps,frame_id):
        # possesso palla teams
        if(len(boxes_ball) > 0) or (len(self.last_valid_ball) > 0):
            if(len(boxes_ball) > 0):# a new valid ball position from the det+tracker
                self.last_valid_ball = boxes_ball
            else:   #if the det+tracker doesn't find a ball use the last one position
                boxes_ball = self.last_valid_ball
                        
            ball_players_distance = []        

            for box in boxes_team:
                ball_players_distance.append(distance_boxes(box,boxes_ball[0]))
            
            player_index = np.argmin(ball_players_distance)

            txt = "-"

            if(ball_players_distance[player_index] < 150):
                
                team_number = int(team_numbers[player_index])
                
                self.storia_possesso_palla.append(team_number)
                
                #the current team number is defined as the most recurrent number in the last 5 frame
                self.filtered_team_number = int(np.median(self.storia_possesso_palla[-10:]))
                
                self.possesso_palla[self.filtered_team_number] = self.possesso_palla[self.filtered_team_number] + 1
                
                image = draw_rect(image, boxes_team[player_index], (0,0,255))
                
                circle_player(image, boxes_team[player_index], 150)
                
                if (self.filtered_team_number) == 0:
                    txt = "Arbitro"
                if (self.filtered_team_number) == 1:
                    txt = "Team 1"
                if (self.filtered_team_number) == 2:
                    txt = "Team 2"

                cv2.putText(
                    image, #numpy array on which text is written
                    txt, #text
                    (int(boxes_team[player_index][0]), int(boxes_team[player_index][1] - 60)), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    1, #font size
                    (40, 40, 40, 255), #font color
                    3) #font stroke       
                
            image = cv2.putText(image, "In possesso: {}".format(txt), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,255), 4)

            #print possesso palla delle squadre:  
            image = cv2.putText(image, "Possesso palla (sec/tot)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 4)

            elapsed = round(frame_id / fps, 1)
            #arbitri
            image = cv2.putText(image, "   Arbitri: {}/{} s".format(round(self.possesso_palla[0] / fps, 1), elapsed), (100, 200+(80 * 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
            #Team 1                
            image = cv2.putText(image, "   Team 1: {}/{} s".format(round(self.possesso_palla[1] / fps, 1), elapsed), (100, 200 + (80 * 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)                        
            #Team 2               
            image = cv2.putText(image, "   Team 2: {}/{} s".format(round(self.possesso_palla[2] / fps, 1), elapsed), (100, 200 + (80 * 3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)    
            
        return image

    # Generazione statistica 2
    def stat2(self,image,boxes_ball,line_points_arr):
        #posizione palla metà campo DX o SX
        if(len(boxes_ball) > 0) or (len(self.last_valid_ball) > 0):
            print("Length boxes_ball: {} - Length last: {} ".format(len(boxes_ball), len(self.last_valid_ball)))
            if(len(boxes_ball) > 0):     # a new valid ball position from the det+tracker
                self.last_valid_ball = boxes_ball
            else:   #if the det+tracker doesn't find a ball use the last one position
                boxes_ball = self.last_valid_ball
                
            coor = boxes_ball[0]
            line_points_arr = np.asarray(self.line_points)
            p1 =[coor[0],coor[1]]

            # Return -1 left, 0 on line, +1 right
            ball_pos = ball_position(self.line_points[0], self.line_points[1], p1)

            distance_ball_center = abs((np.cross(line_points_arr[0]-p1, p1-line_points_arr[1])) / np.linalg.norm(line_points_arr[0]-p1))
            self.history_distance_ball_center.append(distance_ball_center * ball_pos)

            self.ballDX = (ball_pos == 1)
            self.ballSX = (ball_pos == -1)

            if ball_pos == 1:
                self.ball_cumulative_position[0] = self.ball_cumulative_position[0] + 1
                image = cv2.putText(image, "Posizione palla: DX", (100, 200 + (80 * 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 4)   
            else:
                self.ball_cumulative_position[1] = self.ball_cumulative_position[1] +1
                image = cv2.putText(image, "Posizione palla: SX", (100, 200 + (80 * 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 4)
        else:
            image = cv2.putText(image, "Posizione palla: -", (100, 200 + (80 * 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 4)  

        return image
    
    # Generazione statistica 4
    def stat4(self,image,boxes_ball,boxes_team,team_numbers):        
        if(len(boxes_ball) > 0):
            attacco = False
                
            ball_team1_distance = []
            ball_team2_distance = []
            
            number_of_value = 10  #number of value for mean
            distance_search = 250   #how far i must search for a crowded frame
                          
            for i, box in enumerate(boxes_team):
                if int(team_numbers[i]) == 1 :
                    ball_team1_distance.append(distance_boxes(box,boxes_ball[0]))
                if int(team_numbers[i]) == 2 :
                    ball_team2_distance.append(distance_boxes(box,boxes_ball[0]))
                
            self.history_mean_dist_team1.append(np.mean(ball_team1_distance))
            self.history_mean_dist_team2.append(np.mean(ball_team2_distance)) 
     
            # with the history of all the players near the ball find the contropiede 
            if  len(self.history_mean_dist_team1) > distance_search:
                #media giocatori vicini alla palla per ogni squadra negli ultimi 5 frame
                mean_team1 = np.mean(self.history_mean_dist_team1[-number_of_value:])
                mean_team2 = np.mean(self.history_mean_dist_team2[-number_of_value:])             
                  
                if (mean_team1 + mean_team2) / 2 > 400: 
                    #print("poco affollato:  mean:" +str((mean_team1+mean_team2)/2))
                    last_50_dist_team1 = self.history_mean_dist_team1[-distance_search:-number_of_value]
                    last_50_dist_team2 = self.history_mean_dist_team2[-distance_search:-number_of_value]                    
                    
                    frame_crowded = 0
                    for i in range(len(last_50_dist_team1)):
                        #verifica se in una posizione passata intorno alla palla c'erano alemeno 6 giocatori                   
                        if((last_50_dist_team1[i] + last_50_dist_team2[i]) / 2) < 300:
                            frame_crowded += 1                        
                        
                    if frame_crowded > number_of_value and (self.filtered_team_number != 2 or self.filtered_team_number != 1):
                        attacco = True
                        #print("number of frame crowded befor a single player action:  "+str(frame_crowded))
                        #print(np.gradient(history_distance_ball_center[-80:]))
                        direction = np.mean(np.gradient(self.history_distance_ball_center[-80:]))

                        #print("Gradient: {}".format(direction))
                        
                        (H, W) = image.shape[:2]
                        
                        image = cv2.rectangle(image, (int((W/2)-200), 50), (int((W/2)+200), 300), (0,0,0), -1)
                        #image = cv2.rectangle(image, (int((W/2)-200), 50), (int((W/2)+200), 300), (150, 50, 50), 5) 
                         
                        cv2.putText(image,"Direction of Attack: ", (int((W/2)-150), 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

                        
                        if direction > 0:
                           # print("attacco a DX")
                            cv2.arrowedLine(image, (int((W/2)), 200), (int((W/2))+150, 200), (200,200,200), 8, tipLength=0.5)
                             
                        if direction < 0:
                            #print("attacco a SX")
                            cv2.arrowedLine(image, (int((W/2)), 200), (int((W/2)-150), 200), (200,200,200), 8,tipLength=0.5)         
        return image
    
    # Generazione statistica 5
    def stat5(self,image):   
        #ricera zona affollata         
        number_of_value = 10  #number of value for mean
        distance_search = 250   #how far i must search for a crowded frame
        
        (H, W) = image.shape[:2]
        
        mean_team1 = np.mean(self.history_mean_dist_team1[-number_of_value:])
        mean_team2 = np.mean(self.history_mean_dist_team2[-number_of_value:])
        image = cv2.putText(image, "Pressione avversaria", (100, 200 + (80 * 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 4)

        if (mean_team1 + mean_team2) / 2 < 500:
           # print("affollato")
            if self.ballSX :
                #print("affollamento a SX")
                self.pressione[0] += 1
                
            if self.ballDX:
                #print("affollamento dx")
                self.pressione[1] += 1
                
        if np.sum(self.pressione) > 0:
            image = cv2.putText(image, "   Team 1: {}%".format(str(int(self.pressione[1] / np.sum(self.pressione) * 100))), (100, 200 + (80 * 6)), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
            image = cv2.putText(image, "   Team 2: {}%".format(str(int(self.pressione[0] / np.sum(self.pressione) * 100))), (100, 200 + (80 * 7)), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        
        return image

    def run_stats(self,image,boxes_ball,boxes_team,team_numbers,fps,frame_id):
        #Draw stats windows
        image = cv2.rectangle(image, (50,50), (700, 100 + (80 * 9)), (0,0,0), -1)  

        #Draw pitch middle line
        image = cv2.line(image, self.line_points[0], self.line_points[1], (20,20,20), thickness=2)

        # Chiamata statistica 1            
        image = self.stat1(image, boxes_ball, boxes_team, team_numbers, fps, frame_id)
        
        # Chiamata statistica 2        
        image = self.stat2(image, boxes_ball, self.line_points)
        
        # Chiamata statistica 4
        image = self.stat4(image, boxes_ball, boxes_team, team_numbers)   
        
        # Chiamata statistica 5
        image = self.stat5(image)

        return image
        

def run_all(video_path, ball_tracking_path, team_detection_path, out_txt_file):
    ball_dict = get_dict(ball_tracking_path)
    team_dict = get_dict(team_detection_path)
       
    # Input video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Output video
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #definisco formato output video mp4
    out = cv2.VideoWriter('output-finale.mp4',fourcc, 30.0, (int(video.get(3)/2), int(video.get(4)/2)), True) #definisco proprietà output video

    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    stat = Statistics()
    
    ret, img = video.read()
    stat.initialize(img, 1)
    
    ret = True   
    frame_id = 1
    
    while ret:
        ret, frame = video.read()
    
        if not ret:
            continue

        image = frame

        #estrazione boxes teams e palla dai rispettivi dataset
        boxes_team, scores_team, names_team, team_numbers = [[0,0,0,0]], [[0]], [[0]], [[0]]
        boxes_team,scores_team,names_team, complete_team, team_numbers = get_gt(frame,frame_id,team_dict)                    

        boxes_ball, scores_ball, names_ball, not_used= [[0,0,0,0]], [[0]], [[0]], [[0]]
        boxes_ball,scores_ball,names_ball, complete_ball, not_used = get_gt(frame,frame_id,ball_dict)
        
        image = draw_players(image,boxes_team,team_numbers)
        if len(boxes_ball) > 0:
            coor = boxes_ball[0] #nel caso della palla il detector ritorna solo 1 detection ogni volta
            draw_rect(image, coor, (10,255,255))
          
        stat.run_stats(image, boxes_ball, boxes_team, team_numbers, fps, frame_id)
             
        image_to_show = cv2.resize(image, (1920, 1080))
        cv2.imshow("image",image_to_show)
        cv2.waitKey(1)
        
        (H, W) = image.shape[:2]
        i = cv2.resize(image, (int(W/2), int(H/2)))
        out.write(i) 
        
        frame_id+=1
        
    out.release()
    
    #output file for statistics
    try:
        os.remove(out_txt_file)
    except :
        None
    f = open(out_txt_file, "a")
    
    stat.generate_file(f, frame_id)
    f.close()
        
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

    #stats(args.video, args.ball_track, args.det_teams, args.out_stats)
    
    run_all(args.video, args.ball_track, args.det_teams, args.out_stats)