
# Automatic statistics generator on basketballs video 

![statistic generation example](stat_example.png)
![structure of the project](project_structure.png)

# To use this code: <h5>
1. Install the maskrcnn requirements (mask_rcnn_UPG_x_playerDetection/requirements.txt    + python setup.py)
2. Install opencv & numpy (per yolo, trackerCSRT)

	NOTE: in the last version of openCV library the tracker CSRT file tracker_parameters.yaml is changed!!!! if there are some problems with this file please remove the lines that use that file in the custom_tracking_v2.py file
	
3. Download the weights files from here   https://drive.google.com/file/d/1Q_wj2b-_jsKw9_PT4BQmtBCT60Zj75cT/view?usp=sharing       and put them in a folder called weights in the directory of the project
4. Download the test video file "prova2" from here    https://drive.google.com/file/d/1s02QRH_yV8yxpA2stwKRzMW_Hf9aeIeW/view?usp=sharing    and put it in a folder called input_video in the directory of the project
5. use the files run_******.sh to run the different part of the framework

NOTE: tests performed on: python 3.8.3, opencv 4.4.0, tensorflow:2.3.1 


# to run the parts of the framework: <h5> 

(look at the image on top of this page to understand the main parts of the framework): 

1. **detetection of the ball with yolo** (execute: run_ball_detector.sh)<br/>
INPUT: video_file<br/>
OUT: file txt in MOT format with the detection positions, output_video<br/>

1. **ball tracking and interpolation** (execute: run_ball_tracker.sh)<br/>	
INPUT: video_file, detection_txt_file<br/>
OUT: file txt with tracking, output_video<br/>

2. **players detection**  (execute: run_player_detection-sh)<br/>
INPUT: video_file<br/>
OUT: file txt with player divided by teams, output_video<br/>

3. **statistics generation**	(execute: run_stats.sh)<br/>
INPUT: video, players_det_txt_file, ball_tracking_file<br/>
OUT: stats.txt, out_video<br/>

If you have doubts about the execution of the .sh files please open them and read the comments and the program call.

Some parts of the project are made in cooperation with Simone.
His project is completely based on Mask R-CNN detector: https://github.com/simoberny/basket_tracking
