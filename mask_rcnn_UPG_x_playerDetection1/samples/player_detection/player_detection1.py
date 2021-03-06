import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
import itertools
import colorsys
import cv2
from time import sleep
from tqdm import tqdm
import math
from PIL import Image
from math import sqrt
import time

from utility.player_utility import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

team_1 = [60,60,60,0]
team_2 = [200,200,200,0]
arbitro = [0, 102, 204, 0]

# define random colors
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

#Take the image and apply the mask, box, and Label
def display_instances(count, image, boxes, masks, ids, names, scores, resize):
    f = open("det/det_player_maskrcnn.txt", "a")

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    color_list = []

    if not n_instances:
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None

        width = x2 - x1
        height = y2 - y1
        
        #If a player
        if score > 0.75 and label == 'person':
            mask = masks[:, :, i]

            #Create a masked image where the pixel not in mask is green
            image_to_edit = image.copy()
            mat_mask = cut_by_mask(image_to_edit, mask)

            offset_w = int(width/6)
            offset_h = int(height/3)
            offset_head = int(height/8)

            #file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(file_name, mat_mask)

            #Crop the image with some defined offset
            crop_img = mat_mask[y1+offset_head:y2-offset_h, x1+offset_w:x2-offset_w]

            #Return one single dominant color
            rgb_color = get_dominant(crop_img)

            #Add to the list of all the bbox color found in the single frame
            color_list.append(rgb_color)

            rgb_tuple = tuple([int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])]) 

            caption = '{} {:.2f}'.format(label, score) if score else label
        
            image = apply_mask(image, mask, rgb_tuple)
            image = cv2.rectangle(image, (x1+offset_w, y1+offset_head), (x2-offset_w, y2-offset_h), rgb_tuple, 2)
            image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, rgb_tuple, 2)

            team = getTeam(image, rgb_color)

            f.write('{},-1,{},{},{},{},{},-1,-1,-1,{}\n'.format(count, x1*resize, y1*resize, (x2 - x1)*resize, (y2 - y1)*resize, score, team))

    #Group to 3 cluster all the color found in the frame's bboxes
    clusters, counts = parse_colors(color_list, 3)

    #Update team's stats
    image = draw_team(image, clusters, counts)

    '''file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, image)'''

    f.close()

    return image

def video_segmentation(model, class_names, video_path, resize=2, display=False):
    start = time.time()

    f = open("det/det_player_maskrcnn.txt", "w").close()
    
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    length_input = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create video writer
    file_name = "output/detection_player_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (int(width/resize), int(height/resize)))
    
    count = 0
    success = True

    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while success:
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Basket pitch mask
                '''
                mask = get_mask('roi_mask.jpg')
                mask = np.expand_dims(mask,2)
                mask = np.repeat(mask,3,2)
                #Apply pitch mask to esclude the people outside
                image = image * mask
                image = image.astype(np.uint8)
                '''

                # Resize for better performance
                image = cv2.resize(image, (int(width/resize), int(height/resize)))

                #Detect objects
                r = model.detect([image], verbose=0)[0]

                #Process objects
                frame = display_instances(count, image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"], resize)
                # RGB -> BGR to save image to video
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if display:
                    cv2.imshow('MaskRCNN Ball Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Add image to video writer
                vwriter.write(frame)
                count += 1

            #Needed per the print progress
            pbar.update(1)
            sleep(0.01)

    vwriter.release()

    end = time.time()
    print("Saved to ", file_name)

    print("Detections time: ", end-start)
    print("FPS: {}".format(length_input/(end-start)))
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("--command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-d', '--display', required=False, action='store_true')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights


    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

        model.load_weights(weights_path, by_name=True)
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "detect":
        video_segmentation(model, class_names, video_path=args.video, display=args.display)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
