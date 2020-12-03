from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup
import IPython
import os
import json
import random
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import base64
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from object_detection.utils import label_map_util

class QuickDrawDataset:
    def __init__(self):
        self.qd = QuickDrawData()
        self.coco_to_quickdraw = {
            "pizza": "pizza",
            "bus": "bus",
            "couch": "couch",
            "remote": "key",
            "snowboard": "skateboard",
            "fork": "fork",
            "suitcase": "suitcase",
            "scissors": "scissors",
            "sheep": "sheep",
#             "person": "scorpion",
            "person": "teddy-bear",
            "dog": "dog", 
            "cake": "cake",
            "tv": "television",
            "frisbee": "baseball",
            "carrot": "carrot",
            "bird": "bird",
            "bicycle": "bicycle",
            "cow": "cow",
            "cat": "cat",
            "bowl": "bathtub",
            "book": "book",
            "tie": "bowtie",
            "sink": "sink",
            "toothbrush": "toothbrush",
            "bed": "bed",
            "bear": "bear",
            "skateboard": "skateboard",
            "train": "train",
            "backpack": "backpack",
            "cup": "cup",
            "bottle": "wine bottle",
            "car": "car",
            "laptop": "laptop",
            "elephant": "elephant",
            "bench": "bench",
            "knife": "knife",
            "zebra": "zebra",
            "orange": "apple",
            "donut": "donut",
            "apple": "apple",
            "chair": "chair",
            "toilet": "toilet",
            "banana": "banana",
            "airplane": "airplane",
            "giraffe": "giraffe",
            "refrigerator": "cooler",
            "broccoli": "broccoli", 
            "clock": "clock", 
            "spoon": "fork", 
            "handbag": "purse", 
            "horse": "horse", 
            "toaster": "toaster", 
            "microwave": "microwave", 
            "motorcycle": "motorbike", 
            "vase": "vase", 
            "keyboard": "keyboard", 
            "skis": "lollipop", 
            "oven": "oven", 
            "kite": "postcard", 
            "surfboard": "skateboard", 
            "sandwich": "sandwich", 
            "truck": "truck", 
            "boat": "speedboat", 
            "umbrella": "umbrella", 
            "mouse": "mouse", 
            "traffic light": "traffic light", 
            "fire hydrant": "fire hydrant", 
            "stop sign": "stop sign", 
            "parking meter": "sword", 
            "sports ball": "soccer ball", 
            "baseball bat": "baseball bat", 
            "baseball glove": "hand", 
            "tennis racket": "tennis racquet", 
            "wine glass": "wine glass", 
            "hot dog": "hot dog", 
            "potted plant": "house plant", 
            "dining table": "table", 
            "cell phone": "cell phone", 
            "teddy bear": "teddy-bear", 
            "hair dryer": "drill" }
      
        PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS, use_display_name=True)
        
    # Returns smallest axis-aligned rectangle containing points. Input points are in fractional (Object Detection API) format.
    # Output box is in pixel (COCO dataset annotation) format.
    def surrounding_box(self, points, image_shape):
        min_x = min([point[0] for point in points])
        max_x = max([point[0] for point in points])
        min_y = min([point[1] for point in points])
        max_y = max([point[1] for point in points])
        
        surrounding_bbox = [min_y * image_shape[1],
                    min_x * image_shape[0],
                    (max_y - min_y) * image_shape[1],
                    (max_x - min_x) * image_shape[0]]
        
        return surrounding_bbox
        
        

    # Removes detections that either do not have a quickdraw equivalent annotations or have a score
    # below the threshold. Returns pruned detections, used for drawing, and mask detections, used
    # for masking.
    def prune_detections(self, image_shape, detections, threshold):
        pruned_detections = []
        mask_detections = []
        body_detections = []
        for i in range(int(detections['num_detections'][0])):
            if ((self.category_index[detections['detection_classes'][0][i]]['name'] in self.coco_to_quickdraw) and
                (detections['detection_scores'][0][i] > threshold)):
                old_bbox = detections['detection_boxes'][0][i]
                quickdraw_class = self.coco_to_quickdraw[self.category_index[detections['detection_classes'][0][i]]['name']]
                new_bbox = [old_bbox[1] * image_shape[1],
                            old_bbox[0] * image_shape[0],
                            (old_bbox[3] - old_bbox[1]) * image_shape[1],
                            (old_bbox[2] - old_bbox[0]) * image_shape[0]]
                mask_detections.append({'class': quickdraw_class,'bbox': new_bbox})
                    
                if self.category_index[detections['detection_classes'][0][i]]['name'] == 'person':
                    # Head
                    head_keypoints = [0, 1, 2, 3, 4, 5, 6] # nose, left/right eyes, left/right ears, left/right shoulder
                    head_points = [detections['detection_keypoints'][0][i][j] for j in head_keypoints]
#                     head_top = old_bbox[1]
#                     head_bottom = max(detections['detection_keypoints'][0][i][5][1],
#                                    detections['detection_keypoints'][0][i][6][1])
#                     print('head top+bot: ', head_top, head_bottom)
#                     head_bbox = self.surrounding_box(head_points, image_shape, head_top, head_bottom)
                    head_bbox = self.surrounding_box(head_points, image_shape)
                    pruned_detections.append({'class': 'face', 'bbox': head_bbox})
                    body_detections.append({'class': 'face', 'bbox': head_bbox})
                    
                    # Torso
                    torso_keypoints = [5, 6, 7, 8, 11, 12] # left/right shoulder, left/right elbow, left/right hip
                    torso_points = [detections['detection_keypoints'][0][i][j] for j in torso_keypoints]
                    torso_bbox = self.surrounding_box(torso_points, image_shape)
                    pruned_detections.append({'class': 't-shirt', 'bbox': torso_bbox})
                    body_detections.append({'class': 't-shirt', 'bbox': torso_bbox})
                    
                    # Legs
                    legs_keypoints = [11, 12, 13, 14, 15, 16] # left/right hip, left/right knee, left/right ankle
                    legs_points = [detections['detection_keypoints'][0][i][j] for j in legs_keypoints]
                    legs_bbox = self.surrounding_box(legs_points, image_shape)
                    pruned_detections.append({'class': 'pants', 'bbox': legs_bbox})
                    body_detections.append({'class': 'pants', 'bbox': legs_bbox})
                else: 
                    pruned_detections.append({'class': quickdraw_class,'bbox': new_bbox})
    
        return pruned_detections, mask_detections, body_detections
    
    # get dimensions of drawing
    def get_quickdraw_dims(self, drawing):
        smallest_x = float('inf')
        smallest_y = float('inf')
        largest_x = 0
        largest_y = 0

        for stroke in drawing.strokes:
            xarr = []
            yarr = []

            for x, y in stroke:
                xarr.append(x)
                yarr.append(y)
 
                if(x <= smallest_x):
                    smallest_x = x
                if(y <= smallest_y):
                    smallest_y = y
                if(x >= largest_x):
                    largest_x = x
                if(y >= largest_y):
                    largest_y = y

        width = largest_x-smallest_x
        height = largest_y - smallest_y

        return width, height

    
    def draw_blank(self, drawing, image, bbox_x, bbox_y, bbox_w, bbox_h):
        I = np.zeros(image.shape)
        width, height = self.get_quickdraw_dims(drawing)

        for stroke in drawing.strokes:
            xarr = []
            yarr = []
            for x, y in stroke:
                xarr.append(x)
                yarr.append(y)

            newx = [x * (bbox_w/width) + bbox_x for x in xarr]
            newy = [y * (bbox_h/height) + bbox_y for y in yarr]
            newx = np.rint(newx).astype(int)
            newy = np.rint(newy).astype(int)
            for i in range(0, len(newx) - 1):
                I = cv2.line(I, (newx[i],newy[i]), (newx[i+1],newy[i+1]), (255, 255, 255) , 1)
        return I
    
    # Returns nearest neighbor to image (inv) from quick draw data group (qdg) of the same class.
    # Adds augmented (rotated and flipped images)
    def nearest_neighbor(self, qdg, inv, bbox_x, bbox_y, w, h, logging=False):
        smallest_norm = float('inf')
        closest_drawing = qdg.get_drawing()

        for drawing in qdg.drawings:
            drawn_image = self.draw_blank(drawing, inv, bbox_x, bbox_y, w, h)
            drawn_image = cv2.GaussianBlur(drawn_image,(5,5),cv2.BORDER_DEFAULT)
#             drawn_image = cv2.bitwise_not(drawn_image)
            fnorm = np.linalg.norm(drawn_image - inv, 'fro')


            if(fnorm < smallest_norm):
                if logging:
                    plt.imshow(drawn_image, cmap='Greys')
                    plt.axis('off')
                    plt.show() 
                smallest_norm = fnorm
                closest_drawing = drawing.key_id

        final = qdg.search_drawings(key_id=closest_drawing)[0]

        return final
    
    # Draws pruned detections on image.
    def draw(self, image, orig_image, detections, save_filename, logging=False):
        I = image.copy()
        Im = orig_image.copy()

        for detection in detections:

            plt.axis('off')

            [bbox_x, bbox_y, bbox_w, bbox_h] = detection['bbox']

            bbox_x = int(bbox_x)
            bbox_y = int(bbox_y)
            bbox_w = int(bbox_w)
            bbox_h = int(bbox_h)

            bbox_image = Im[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            edges = cv2.Canny(bbox_image, 50, 400, apertureSize = 3, L2gradient = True)
            inv = edges
#             inv = cv2.bitwise_not(edges)
#             print('inv shape', inv.shape)
            h, w = np.shape(inv)
            masked_bbox_image = np.full((I.shape[0], I.shape[1]), 0, dtype=float)
#             print('exp shape', masked_bbox_image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w].shape)
            masked_bbox_image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = inv
#             print('masked_bbox_image')
            masked_bbox_image = cv2.GaussianBlur(masked_bbox_image,(5,5),cv2.BORDER_DEFAULT)
    
            if logging:
                plt.imshow(masked_bbox_image, cmap='Greys')
                plt.axis('off')
                plt.show()   

            qdg = QuickDrawDataGroup(detection['class'], recognized=True)

#             final = qdg.get_drawing()
            final = self.nearest_neighbor(qdg, masked_bbox_image, bbox_x, bbox_y, w, h, logging=True)
#             final = cv2.bitwise_not(final)
            
            width, height = self.get_quickdraw_dims(final)

            for stroke in final.strokes:
                xarr = []
                yarr = []
                for x, y in stroke:
                    xarr.append(x)
                    yarr.append(y)

                newx = [x * (bbox_w/width) + bbox_x for x in xarr]
                newy = [y * (bbox_h/height) + bbox_y for y in yarr]
                newx = np.rint(newx).astype(int)
                newy = np.rint(newy).astype(int)
                for i in range(0, len(newx) - 1):
                    I = cv2.line(I, (newx[i],newy[i]), (newx[i+1],newy[i+1]), (255, 255, 255) , 2)

        if logging:
            plt.imshow(I)
            plt.axis('off')
            plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
            plt.show()                                                
            
        
        return I