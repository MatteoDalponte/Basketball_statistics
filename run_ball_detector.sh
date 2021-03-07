python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball832x832.cfg -w weights/yolov3_ball_5000_832x832_augmented.weights -v input_video/prova2.mp4 -s -out_txt det_tracking/prova2_aug1.txt

#yolo con al prima versione del dataset (non aumentato e con grandeza 416x416)
#python yolo_det.py -l weights/obj.names -cfg weights/yolov3_ball.cfg -w weights/yolov3_2000_round2.weights -v input_video/prova2.mp4 -s -out_txt det_tracking/prova2_no_aug_416.txt
