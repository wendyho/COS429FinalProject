from object_detection_model import ObjectDetectionModel
from inpaint import Inpainting
from quick_draw_dataset import QuickDrawDataset
from util import resize
from util import load_image_into_numpy_array

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Runs image evaluation on image in source_path and saves results to sink_path
def evaluate(object_detection, qd_dataset, inpainting, source_path, sink_path):
    for filename in os.listdir(source_path):
        if filename.endswith(".JPG") or filename.endswith(".PNG") or filename.endswith(".jpeg") or filename.endswith(".jpg"):
            print('Evaluating : ', filename)
            image_np = load_image_into_numpy_array(os.path.join(source_path, filename))
            image_resized = resize(image_np, 256)
            
            detections = object_detection.inference(image_resized, logging=True)
            pruned_detections, mask_detections, body_detections = qd_dataset.prune_detections(image_resized.shape, detections, threshold=0.4)
            
            _ = inpainting.create_mask(body_detections, logging=True)
            mask = inpainting.create_mask(mask_detections, logging=True)
            image_inpainted = inpainting.inpaint(image_resized, mask, logging=True)
        
            
            drawn_image = qd_dataset.draw(image_inpainted,image_resized, pruned_detections, os.path.join(sink_path, filename), logging=True)
#             qd_dataset.plot_histogram()
            
if __name__ == "__main__":
    object_detection = ObjectDetectionModel()
    qd_dataset = QuickDrawDataset()
    inpainting = Inpainting()
    evaluate(object_detection, qd_dataset, inpainting, '../data/example_images', '../data/example_images_drawn')