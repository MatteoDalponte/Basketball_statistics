
Automatic statistics generator on basketballs video 


The project is made by separate steps:

1) detetection of the ball with yolo (run main_detection.py)  INPUT: video_file     			OUT: file txt in MOT format with the detection positions, output_video
2) ball tracking and interpolation (run run_tracker.txt)	INPUT: video_file, detection_txt_file		OUT: file txt with tracking, output_video
3) players detections						INPUT: video_file 				OUT: file txt with player divided by teams, output_video
4) statistics			(run_stats.sh)			INPUT: video, players det txt file, ball tracking file	OUT: stats.txt, out_video
