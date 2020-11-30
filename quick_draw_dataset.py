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

    # Removes detections that either do not have a quickdraw equivalent annotations or have a score
    # below the threshold.
    def prune_detections(self, image_shape, detections, threshold):
        pruned_detections = []
        for i in range(int(detections['num_detections'][0])):
            if ((self.category_index[detections['detection_classes'][0][i]]['name'] in self.coco_to_quickdraw) and
                (detections['detection_scores'][0][i] > threshold)):
                quickdraw_class = self.coco_to_quickdraw[self.category_index[detections['detection_classes'][0][i]]['name']]
                old_bbox = detections['detection_boxes'][0][i]
                new_bbox = [old_bbox[1] * image_shape[1],
                            old_bbox[0] * image_shape[0],
                            (old_bbox[3] - old_bbox[1]) * image_shape[1],
                            (old_bbox[2] - old_bbox[0]) * image_shape[0]]
                pruned_detections.append({'class': quickdraw_class,'bbox': new_bbox})
    
        return pruned_detections
    
    # get dimensions of drawing
    def get_quickdraw_dims(drawing):
        smallest_x = float('inf')
        smallest_y = float('inf')
        largest_x = 0
        largest_y = 0

        for stroke in drawing.strokes:
            xarr = []
            yarr = []
        #    plt.plot(stroke, marker = 'o')

            for x, y in stroke:
                xarr.append(x)
                yarr.append(y)
                #plt.plot(x, y, marker = ',')
                #print("x={} y={}".format(x, y))
                if(x <= smallest_x):
                    smallest_x = x
                if(y <= smallest_y):
                    smallest_y = y
                if(x >= largest_x):
                    largest_x = x
                if(y >= largest_y):
                    largest_y = y

        #     plt.plot(xarr,yarr, marker = ',', color="white")

        width = largest_x-smallest_x
        height = largest_y - smallest_y

        return width, height
    
    # return nearest neighbor to image (inv) from quick draw data group (qdg) of the same class
    def nearest_neighbor(qdg, inv, w, h):
        smallest_norm = float('inf')
        closest_drawing = 0

        for drawing in qdg.drawings:
            pix = np.array(drawing.image)
            pixgray = cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY)
            pixgray = cv2.resize(pixgray, (w, h))
            fnorm = np.linalg.norm(pixgray - inv, 'fro')

            if(fnorm < smallest_norm):
                smallest_norm = fnorm
                closest_drawing = drawing.key_id

        final = qdg.search_drawings(key_id=closest_drawing)[0]

        return final
    
    # Draws pruned detections on image.
    def draw(self, image, orig_image, detections, save_filename, logging=False):
        I = image.copy()
        Im = orig_image.copy()

        for detection in detections:
            
#             qd = QuickDrawData()
#             drawing = self.qd.get_drawing(detection['class'])
#             drawing = qd.get_drawing("bus")

            plt.axis('off')

            [bbox_x, bbox_y, bbox_w, bbox_h] = detection['bbox']

            bbox_x = int(bbox_x)
            bbox_y = int(bbox_y)
            bbox_w = int(bbox_w)
            bbox_h = int(bbox_h)

            bbox_image = Im[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            edges=cv2.Canny(bbox_image, 150, 200)
            inv = cv2.bitwise_not(edges)

            h, w = np.shape(inv)

            qdg = QuickDrawDataGroup(detection['class'], recognized=True)

            final = QuickDrawDataset.nearest_neighbor(qdg, inv, w, h)

            width, height = QuickDrawDataset.get_quickdraw_dims(final)

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

        plt.imshow(I)
        plt.axis('off')
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
        plt.show()                                                
            
#             drawing = self.qd.get_drawing(detection['class'])
#             pil_im = drawing.image

#             plt.axis('off')

#             smallest_x = float('inf')
#             smallest_y = float('inf')
#             largest_x = 0
#             largest_y = 0
#             for stroke in drawing.strokes:
#                 xarr = []
#                 yarr = []

#                 for x, y in stroke:
#                     xarr.append(x)
#                     yarr.append(y)
#                     if(x <= smallest_x):
#                         smallest_x = x
#                     if(y <= smallest_y):
#                         smallest_y = y
#                     if(x >= largest_x):
#                         largest_x = x
#                     if(y >= largest_y):
#                         largest_y = y

#             width = largest_x-smallest_x
#             height = largest_y - smallest_y

#             [bbox_x, bbox_y, bbox_w, bbox_h] = detection['bbox']
#             for stroke in drawing.strokes:
#                 xarr = []
#                 yarr = []
#                 for x, y in stroke:
#                     xarr.append(x)
#                     yarr.append(y)

#                 newx = [x * (bbox_w/width) + bbox_x for x in xarr]
#                 newy = [y * (bbox_h/height) + bbox_y for y in yarr]
#                 plt.plot(newx,newy, marker = ',', color="white")
                
#         if logging:
#             plt.imshow(I)
#             plt.axis('off')
#             plt.show()
        
        return I