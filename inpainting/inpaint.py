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

from core.utils import set_device, postprocess, set_seed

# image path the path to the image to read
# mask will be a path to an image as well with 255 in the mask locations
def inpaint(image_path, mask_path=None):
    # torch.cuda.set_device(0)
    set_seed(2020)

    # Model and version
    model_dir = 'pen_inpaint_model'
    net = importlib.import_module('model.pennet')
    model = set_device(net.InpaintGenerator())
    latest_epoch = open(os.path.join(model_dir, 'latest.ckpt'), 'r').read().splitlines()[-1]
    path = os.path.join(model_dir, 'gen_{}.pth'.format(latest_epoch))
    data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
    model.load_state_dict(data['netG'])
    model.eval()

    path = 'inpaint_output'
    os.makedirs(path, exist_ok=True)
    
    if mask_path  == None: 
        mask = np.zeros((256, 256)).astype(np.uint8)
        mask[256//4:256*3//4, 256//4:256*3//4] = 255 
        mask = Image.fromarray(mask).convert('L')
        masks = F.to_tensor(mask)[None,:,:]
    else: 
        mask = Image.open(mask_path).convert('L')
        masks = F.to_tensor(mask)[None, :,:]
        print(masks.shape)
    
    img = Image.open(image_path).convert('RGB')
    img = F.to_tensor(img) * 2 -1.
    images = img[None, :,:,:]

    print('Inpainting...')
    images, masks = set_device([images, masks])
    images_masked = images*(1-masks) + masks
    with torch.no_grad():
        _, output = model(torch.cat((images_masked, masks), dim=1), masks)
    orig_imgs = postprocess(images)
    mask_imgs = postprocess(images_masked)
    comp_imgs = postprocess((1-masks)*images+masks*output)
    pred_imgs = postprocess(output)
    for i in range(len(orig_imgs)):
        Image.fromarray(pred_imgs[i]).save(os.path.join(path, 'output_pred.png'))
        Image.fromarray(orig_imgs[i]).save(os.path.join(path, 'output_orig.png'))
        Image.fromarray(comp_imgs[i]).save(os.path.join(path, 'output_comp.png'))
        Image.fromarray(mask_imgs[i]).save(os.path.join(path, 'output_mask.png'))
    print('Finish in {}'.format(path))



if __name__ == '__main__':
    inpaint('../../datazip/places2/temp/Places365_val_00000054.jpg', '../../datazip/mask.png')

