# -*- coding: utf-8 -*-

import torch

from torchvision import transforms

import torchvision.transforms.functional as F

from PIL import Image
import numpy as np
import os
import copy
import importlib
import datetime
import matplotlib.pyplot as plt

from inpainting.core.utils import set_device, postprocess, set_seed
# import .model.pennet as net

class Inpainting:
    def __init__(self):
        set_seed(2020)

        # Model and version
        model_dir = os.path.join('inpainting', 'pen_inpaint_model')
        net = importlib.import_module('inpainting.model.pennet')
        self.model = set_device(net.InpaintGenerator())
        latest_epoch = open(os.path.join(model_dir, 'latest.ckpt'), 'r').read().splitlines()[-1]
        path = os.path.join(model_dir, 'gen_{}.pth'.format(latest_epoch))
        data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
        self.model.load_state_dict(data['netG'])
        self.model.eval()
        
    # Image and mask should both be images of size 256 x 256.
    def inpaint(self, img, mask, logging=False):
        masks = F.to_tensor(mask)[None, :,:]
        img = F.to_tensor(img) * 2 -1.
        images = img[None, :,:,:]

        print('Inpainting...')
        images, masks = set_device([images, masks])
        images_masked = images*(1-masks) + masks
        with torch.no_grad():
            _, output = self.model(torch.cat((images_masked, masks), dim=1), masks)
#         orig_imgs = postprocess(images)
#         mask_imgs = postprocess(images_masked)
#         pred_imgs = postprocess(output)
        comp_imgs = postprocess((1-masks)*images+masks*output)

        if logging:
            plt.figure(figsize=(5,5))
            plt.imshow(comp_imgs[0])
            plt.show()

        return comp_imgs[0]

    def create_mask(self, pruned_detections, logging=False):
        mask = np.zeros((256, 256)).astype(np.uint8)

        for detection in pruned_detections:
            bbox = detection['bbox']
            y_dims = (int(bbox[1]), int(bbox[1]+bbox[3]))
            x_dims = (int(bbox[0]), int(bbox[0]+bbox[2]))
            mask[y_dims[0]:y_dims[1], x_dims[0]:x_dims[1]] = 255

        if logging:
            plt.figure(figsize=(5,5))
            plt.imshow(mask)
            plt.show()

        return mask


