#python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v prova2.mp4 -s -out_txt det_tracking/det_yolo.txt
#python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v nba.mp4 -s -out_txt det_tracking/det_yolo_nba.txt
#python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v primo_tempo.mp4 -s -out_txt det_tracking/det_yolo_primo_tempo.txt
#python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v quarto_tempo.mp4 -s -out_txt det_tracking/det_yolo_quarto_tempo.txt
python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v video_alto_part.mp4 -s -out_txt det_tracking/det_yolo_alto_tempo.txt

