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

def get_mask(filename):
	mask = cv2.imread(filename,0)
	mask = mask / 255.0
	return mask
 
#apply mask to image
def apply_mask(image, mask, color, alpha=0.7):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1-alpha) + alpha * c, image[:, :, n])
    
    return image

#apply mask to image
def cut_by_mask(image, mask, color=(0,255,0)):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n], c)
    
    return image

def get_dominant(img):
    global arbitro
    data = np.reshape(img, (-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,labels,palette = cv2.kmeans(data,5,None,criteria,10,flags)
    _, counts = np.unique(labels, return_counts=True)

    best_palette = []
    best_count = 0

    for i, c in enumerate(palette):
        diff = np.sum(np.absolute(c[0:3] - arbitro[0:3]))

        #print("iter {}: {} with {} counts".format(i, c, counts[i]))
        if (c.astype(np.uint8)[1] >= 250 and c.astype(np.uint8)[0] < 15 and c.astype(np.uint8)[2] < 15) or (c.astype(np.uint8) <= 30).all():
            continue
        elif diff < 150 and c[2] > 80: 
            best_palette = np.asarray(arbitro[0:3])
            break
        else:
            if counts[i] > best_count: 
                best_count = counts[i]
                best_palette = c

    return best_palette.astype(np.uint8)

def parse_colors(lst, n):
    cluster = []
    image_array = np.reshape(lst, (len(lst), 3))
    data = np.float32(image_array)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,labels,palette = cv2.kmeans(data,n,None,criteria,10,flags)
    _, counts = np.unique(labels, return_counts=True)

    return palette, counts

def draw_team(image, clusters, counts):
    global team_1
    global team_2
    global arbitro

    image = cv2.rectangle(image, (50,50), (600, 320), (150, 50, 50), 5)

    fusion = []
    for n, clu in enumerate(clusters):
        fusion.append(np.concatenate((clu, int(counts[n])), axis=None))

    fusion = np.array(fusion).astype(np.uint8)

    '''if team_1 == [] and team_2 == []:
        sorted_color = fusion[fusion[:,3].argsort()]

        squadre = sorted_color[1:]

        team_1 = np.append(squadre[0], 1)
        team_2 = np.append(squadre[1], 2)'''

    for el in fusion: 
        diff_0 = np.sum(np.absolute(el[0:3] - team_1[0:3]))
        diff_1 = np.sum(np.absolute(el[0:3] - team_2[0:3]))
        diff_2 = np.sum(np.absolute(el[0:3] - arbitro[0:3]))

        if diff_0 < diff_1 and diff_0 < diff_2:
            team_1[3] = el[3]
        elif diff_1 < diff_0 and diff_1 < diff_2:
            team_2[3] = el[3]
        elif diff_2 < diff_0 and diff_2 < diff_1:
            arbitro[3] = el[3]

    #Arbitro 
    color = tuple([int(arbitro[0]), int(arbitro[1]), int(arbitro[2])])
    image = cv2.rectangle(image, (80,80), (140, 140), color, -1)
    image = cv2.putText(image, "Arbitri ({})".format(arbitro[3]), (180, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)

    #Team 1
    color = tuple([int(team_1[0]), int(team_1[1]), int(team_1[2])])
    image = cv2.rectangle(image, (80,80 + (80 * 1)), (140, 140 + (80 * 1)), color, -1)
    image = cv2.putText(image, "Team {} ({} player)".format(1, team_1[3]), (180, 120 + (80 * 1)), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)
            
    #Team 2
    color = tuple([int(team_2[0]), int(team_2[1]), int(team_2[2])])
    image = cv2.rectangle(image, (80,80 + (80 * 2)), (140, 140 + (80 * 2)), color, -1)
    image = cv2.putText(image, "Team {} ({} player)".format(2, team_2[3]), (180, 120 + (80 * 2)), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)

    return image

def getTeam(image, color):
    global team_1
    global team_2
    global arbitro

    diff_0 = np.sum(np.absolute(color[0:3] - team_1[0:3]))
    diff_1 = np.sum(np.absolute(color[0:3] - team_2[0:3]))
    diff_2 = np.sum(np.absolute(color[0:3] - arbitro[0:3]))

    ret = -1

    if diff_0 < diff_1 and diff_0 < diff_2:
        ret = 1
    elif diff_1 < diff_0 and diff_1 < diff_2:
        ret = 2
    elif diff_2 < diff_0 and diff_2 < diff_1:
        ret = 0

    return ret

#Take the image and apply the mask, box, and Label
def display_instances(count, image, boxes, masks, ids, names, scores):
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

            #Crop the image with some defined offset
            crop_img = mat_mask[y1+offset_head:y2-offset_h, x1+offset_w:x2-offset_w]
            
            '''PIL_image = Image.fromarray(crop_img.astype('uint8'), 'RGB')
            PIL_image.thumbnail((128, 128),Image.ANTIALIAS)'''

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

            f.write('{},-1,{},{},{},{},{},-1,-1,-1,{}\n'.format(count, x1, y1, x2 - x1, y2 - y1, score, team))

    #Group to 3 cluster all the color found in the frame's bboxes
    clusters, counts = parse_colors(color_list, 3)

    #Update team's stats
    image = draw_team(image, clusters, counts)

    '''file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, image)'''

    f.close()

    return image

def video_segmentation(model, class_names, video_path):
    f = open("det/det_player_maskrcnn.txt", "w").close()
    
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    length_input = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create video writer
    file_name = "output/detection_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (width, height))
    
    count = 0
    success = True

    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while success:
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mask = get_mask('roi_mask.jpg')
                mask = np.expand_dims(mask,2)
                mask = np.repeat(mask,3,2)

                #Apply pitch mask to esclude the people outside
                image = image * mask
                image = image.astype(np.uint8)

                #Detect objects
                r = model.detect([image], verbose=0)[0]

                #Process objects
                frame= display_instances(count, image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])
                # RGB -> BGR to save image to video
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Add image to video writer
                vwriter.write(frame)
                count += 1

            #Needed per the print progress
            pbar.update(1)
            sleep(0.01)

    vwriter.release()

    print("Saved to ", file_name)
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
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


    class_names = ['BG', 'basketball']

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
        video_segmentation(model, class_names, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))