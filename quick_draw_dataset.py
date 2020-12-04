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
import pickle
from util import resize

class QuickDrawDataset:
    def __init__(self):
        self.qd = QuickDrawData(recognized = True)
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
            "hair dryer": "drill",
            # dummies to force caluation of sift on person subfeatures
            "xxpersondummy": "face", 
            "xpersondummy2": "t-shirt",
            "xpersondummy3": "pants"}

        PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS, use_display_name=True)

        self.keypoint_dict = self.load_sift_dict(compute=False)

    def keypoint_to_tuple(self, point): 
        return (point.pt, point.size, point.angle, point.response, point.octave, point.class_id) 
    def tuple_to_keypoint(self, point): 
        return cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
                            _response=point[3], _octave=point[4], _class_id=point[5]) 

    # load the sift dictionary, if compute is true, recomputes the sift features
    # how to use the dictionary: dict maps from category -> image
    # loop through images within category, finding best matches + inliers in ransac,
    # take the image with most inliers, homographize the boi
    def load_sift_dict(self, compute=False): 
        # self.qd
        # self.coco_to_quickdraw
        directory = 'quickdraw_keypoints'
        path = directory + '/keypoints.p'
        if not compute and os.path.exists(path):
            print('sifted keypoints found')
            with open(path, 'rb') as fp:
                keypoint_dict = pickle.load(fp)
            # translate back to cv2 keypoints 
            print('translating to keypoints')
            for category, points in keypoint_dict.items(): 
                for i in range(len(points)): 
                    kp_kp = [self.tuple_to_keypoint(points[i][2][j]) for j in range(len(points[i][2]))]
                    points[i] = (points[i][0], points[i][1], kp_kp, points[i][3])
            print('success')
            return keypoint_dict 
            
        print('creating sift keypoint dataset')
        if not os.path.exists(directory):
            os.mkdir('quickdraw_keypoints')


        keypoint_dict_save = dict()
        keypoint_dict = dict()
        # todo do for person categories too, later. 
        sift = cv2.SIFT_create()#contrastThreshold= 0.01)#, edgeThreshold = 1000)
        for _, category in self.coco_to_quickdraw.items(): 
            temp_save = []
            temp = []
            for drawing in self.qd.get_drawing_group(category).drawings: 

                # print(drawing)
                w,h = self.get_quickdraw_dims(drawing)
                I = np.zeros((h,w)) # switch h/w -> row/cols
                I = self.draw_blank(drawing, I, 0, 0, w, h)[:,:, None]
                I = np.array(I, dtype=np.uint8)
                I = cv2.GaussianBlur(I,(3,3),0, borderType=cv2.BORDER_CONSTANT)

                # plt.figure(figsize=(5,5))
                # plt.imshow(I, cmap='Greys')
                # plt.show()
                kp, des = sift.detectAndCompute(I, None)
                tuple_kp = [self.keypoint_to_tuple(kp[i]) for i in range(len(kp))] # save kp as tuples

                # save name just in case, probably unneeded
                temp_save.append((drawing.name, drawing.key_id, tuple_kp,des))
                temp.append((drawing.name, drawing.key_id, kp, des))


                # I = cv2.drawKeypoints(I, kp, I, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
                # plt.figure(figsize=(5,5))
                # plt.imshow(I)
                # plt.show()
                return None
            keypoint_dict_save[category] = temp_save 
            keypoint_dict[category] = temp 
            return 
        print('dumping')
        with open(path, 'wb') as fp: 
            pickle.dump(keypoint_dict_save, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('success')
        return keypoint_dict
        
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
            drawn_image = cv2.GaussianBlur(drawn_image,(5,5),0,cv2.BORDER_DEFAULT)
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
    # get a translation matrix with dx and dy as the translations
    def translateMatrix(self, dx, dy):
        trans = np.diag(np.ones(3))
        trans[0,2] = dx
        trans[1,2] = dy
        return trans
    def getCornersTranslation(self, H, im):
        topl = H @ np.append([0,0],1)
        topl = (topl/topl[2])[:2]
        topr = H @ np.append([0,im.shape[1]-1],1)
        topr = (topr/topr[2])[:2]
        botl = H @ np.append([im.shape[0]-1,0],1)
        botl = (botl/botl[2])[:2]
        botr = H @ np.append([im.shape[0]-1,im.shape[1]-1],1)
        botr = (botr/botr[2])[:2]
        maxx = max(topl[0], topr[0], botl[0], botr[0])
        minx = min(topl[0], topr[0], botl[0], botr[0])
        maxy = max(topl[1], topr[1], botl[1], botr[1])
        miny = min(topl[1], topr[1], botl[1], botr[1])
        
        #find translation matrix
        trans = self.translateMatrix(-minx, -miny)
        # new shape of the transformed image
        new_shape = (int(maxx-minx),int(maxy-miny))

        # find offsets for both images to combine
        # two cases: if translation was in a positive direction then we need to 
        # compensate by moving the static image
        # else if translation was in a negative direction then we need to 
        # compensate by moving the warped image
        # xoff_l = 0
        # yoff_l = 0
        # xoff_r = 0
        # yoff_r = 0
        # if trans[0,2] >= 0:
        #     xoff_r = int(trans[0,2])
        # else:
        #     xoff_l = -1 * int(trans[0,2])
        # if trans[1,2] >= 0:
        #     yoff_r = int(trans[1,2])
        # else: 
        #     yoff_l = -1 * int(trans[1,2])
        return trans, new_shape #, (xoff_l, yoff_l), (xoff_r, yoff_r)

    def draw_sift(self, image, orig_image, detections, save_filename, logging=False):
        I = image.copy()
        Im = orig_image.copy()

        sift = cv2.SIFT_create()
        MIN_MATCH_COUNT = 10
        for detection in detections:
            print(detection)

            plt.axis('off')

            [bbox_x, bbox_y, bbox_w, bbox_h] = detection['bbox']

            bbox_x = int(bbox_x)
            bbox_y = int(bbox_y)
            bbox_w = int(bbox_w)
            bbox_h = int(bbox_h)
            bbox_image = Im[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

            # bbox_image = resize(bbox_image,256)
            edges = cv2.Canny(bbox_image, 50, 400, apertureSize = 3, L2gradient = True)
            inv = edges
            inv = cv2.bitwise_not(edges)
#             print('inv shape', inv.shape)
            h, w = np.shape(inv)
            # masked_bbox_image = np.full((I.shape[0], I.shape[1]), 0, dtype=float)
#             print('exp shape', masked_bbox_image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w].shape)
            # masked_bbox_image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = inv
#             print('masked_bbox_image')

            masked_bbox_image = cv2.GaussianBlur(inv,(5,5),cv2.BORDER_DEFAULT)
            # masked_bbox_image = inv

            print(masked_bbox_image.shape)
            if logging:
                plt.imshow(masked_bbox_image, cmap='Greys')
                plt.show()   
            detect_im = np.array(masked_bbox_image, dtype=np.uint8)
            kp, des = sift.detectAndCompute(detect_im, None)
            detect_im = cv2.drawKeypoints(detect_im, kp, detect_im, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
            plt.figure(figsize=(5,5))
            plt.imshow(detect_im)
            plt.show()

            max_homo = np.identity(3)
            max_inliers = -1
            max_id = -1

            for drawing in self.keypoint_dict[detection['class']]:

                _, draw_id, draw_kp, draw_des = drawing 
                # print(draw_id, draw_kp, draw_des)
                # FLANN_INDEX_KDTREE = 1
                # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                # search_params = dict(checks=50)   
                # flann = cv2.FlannBasedMatcher(index_params,search_params)
                # matches = flann.knnMatch(draw_des,des,k=2)
                bf = cv2.BFMatcher(crossCheck=True)
                matches = bf.match(draw_des, des)

                # print(len(matches))

                # store all the good matches as per Lowe's ratio test.
                # good = []
                # for m,n in matches:
                #     if m.distance < 0.90*n.distance:
                #         good.append(m)
                # print(len(good))
                if len(matches)>MIN_MATCH_COUNT:
                    src_pts = np.array([ draw_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                    dst_pts = np.array([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()
                    numInliers = sum(matchesMask)
                    # print(numInliers)
                    if numInliers > max_inliers: 
                        max_inliers = numInliers
                        max_homo = M 
                        max_id = draw_id

                else:
                    # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
                    matchesMask = None

            if max_inliers < 0: 
                print('not found')
                return 

            # get the drawing 
            qdrawing = self.qd.search_drawings(detection['class'], key_id = max_id)[0]
            print(qdrawing)

            w,h = self.get_quickdraw_dims(qdrawing)
            drawing = np.zeros((h,w)) 
            drawing = self.draw_blank(qdrawing, drawing, 0, 0, w, h)[:,:, None]
            drawing = np.array(drawing, dtype=np.uint8)
            drawing = cv2.GaussianBlur(drawing,(3,3),0, borderType=cv2.BORDER_CONSTANT)

            plt.figure(figsize=(5,5))
            plt.imshow(drawing, cmap='Greys')
            plt.show()
            print(max_homo, max_id, max_inliers)

            trans, new_shape = self.getCornersTranslation(max_homo, drawing)
            print(trans, new_shape)
            warped = cv2.warpPerspective(drawing.T, max_homo, dsize=(new_shape)).T
            print(warped)
            plt.figure(figsize=(5,5))
            plt.imshow(warped, cmap='Greys')
            plt.show()

            # dst = cv2.perspectiveTransform(pts,max_homo)

            # print the warped image onto the original
            # for i in range(bbox_y, b)
            # print(dst)
                    
                        

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

            qdg = self.qd.get_drawing_group(detection['class'])
            # qdg = QuickDrawDataGroup(detection['class'], recognized=True)

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