#the tracker fill the frames left empty by the detector. as input you have to pass the video and the MOTA detection file from the detector.
#The output video will be saved as out-finale.mp4

python custom_tracking_v2.py --det det_tracking/prova2_aug.txt --video input_video/prova2.mp4 --out_tracker det_tracking/tracker_prova2_aug1.txt

