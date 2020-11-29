import tensorflow as tf
import os
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if(path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
          (1, im_height, im_width, 3)).astype(np.uint8)

def resize(im, resize_dim, logging=False):
        resize_ratio = resize_dim / max(im.shape[1], im.shape[2])
        new_size = (int(round(im.shape[2]*resize_ratio)), int(round(im.shape[1]*resize_ratio)))
        new_im = cv2.resize(im[0], new_size)
        delta_h = resize_dim - new_size[1]
        delta_w = resize_dim - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(new_im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        
        if logging:
            plt.figure(figsize=(5,5))
            plt.imshow(new_im)
            plt.show()

        return new_im