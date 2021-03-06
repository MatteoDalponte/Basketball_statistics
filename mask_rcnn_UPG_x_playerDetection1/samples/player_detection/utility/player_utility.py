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
import math
from PIL import Image

# Global teams color initialization
team_1 = [60,60,60,0]
team_2 = [200,200,200,0]
arbitro = [0, 102, 204, 0]

# create binary mask
def get_mask(filename):
	mask = cv2.imread(filename,0)
	mask = mask / 255.0
	return mask
 
# Apply mask to image
def apply_mask(image, mask, color, alpha=0.7):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1-alpha) + alpha * c, image[:, :, n])
    
    return image

# Apply green screen on out of the mask
def cut_by_mask(image, mask, color=(0,255,0)):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n], c)
    
    return image

# Return dominant color from image using Kmeans
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

# Apply Kmeans on a list of color to extract n of it
def parse_colors(lst, n):
    cluster = []
    image_array = np.reshape(lst, (len(lst), 3))
    data = np.float32(image_array)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,labels,palette = cv2.kmeans(data,n,None,criteria,10,flags)
    _, counts = np.unique(labels, return_counts=True)

    return palette, counts

# Draw team color box on image
def draw_team(image, clusters, counts):
    global team_1
    global team_2
    global arbitro

    box_size = (550, 270)
    pad = 30

    (H, W) = image.shape[:2]
    image = cv2.rectangle(image, (pad, (H - box_size[1] - pad)), (pad + box_size[0], (H - pad)), (0, 0, 0), -1)

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

    color_size = 60

    #Arbitro 
    color = tuple([int(arbitro[0]), int(arbitro[1]), int(arbitro[2])])
    image = cv2.rectangle(image, (pad + 30, (H - box_size[1])), (140, (H - box_size[1] + color_size)), color, -1)
    image = cv2.putText(image, "Arbitri ({})".format(arbitro[3]), (180, (H - box_size[1] + color_size - 20)), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)

    #Team 1
    color = tuple([int(team_1[0]), int(team_1[1]), int(team_1[2])])
    image = cv2.rectangle(image, (pad + 30, (H - box_size[1]) + (80 * 1)), (140, (H - box_size[1] + color_size) + (80 * 1)), color, -1)
    image = cv2.putText(image, "Team {} ({} player)".format(1, team_1[3]), (180, (H - box_size[1] + color_size - 20) + (80 * 1)), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)
            
    #Team 2
    color = tuple([int(team_2[0]), int(team_2[1]), int(team_2[2])])
    image = cv2.rectangle(image, (pad + 30, (H - box_size[1]) + (80 * 2)), (140, (H - box_size[1] + color_size) + (80 * 2)), color, -1)
    image = cv2.putText(image, "Team {} ({} player)".format(2, team_2[3]), (180, (H - box_size[1] + color_size - 20) + (80 * 2)), cv2.FONT_HERSHEY_COMPLEX, 1, (200,200,200), 2)

    return image

# Return the team corresponding to the color based on the similarity
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