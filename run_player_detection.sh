
#Note: the file "player_detection1.py)" must be in the maskrcnn/sample follder to run thus:
cd mask_rcnn_UPG_x_playerDetection/samples/player_detection
#Note: the weights and the vido are in the main project folder... go back in the folders to the main with /../../..
python player_detection1.py --weights ../../../weights/mask_rcnn_coco.h5 --video ../../../input_video/prova2.mp4 -d --command detect

#Note: the player detection file with mota benchmark structure is saved in the mask_rcnn_UPG_x_playerDetection/samples/player_detection/det folder.
#To use it statistics python script move it to /det_tracker folder
