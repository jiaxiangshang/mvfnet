import argparse
import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image

import tools
from model import VggEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='/data0/2_Project/python/deeplearning_python/comparison/mvfnet/data/imgs_fs', help='path to load images. It should include image name with: front|left|right')
parser.add_argument('--save_dir', type=str, default='/data0/2_Project/python/deeplearning_python/comparison/mvfnet/data/imgs_fs', help='path to save 3D face shapes')

options = parser.parse_args()
crop_opt = True # change to True if you want to crop the image
imgA = Image.open(os.path.join(options.image_path, 'front.jpg')).convert('RGB')
imgB = Image.open(os.path.join(options.image_path, 'left.jpg')).convert('RGB')
imgC = Image.open(os.path.join(options.image_path, 'right.jpg')).convert('RGB')
if crop_opt:
    imgA = tools.crop_image(imgA)
    imgB = tools.crop_image(imgB)
    imgC = tools.crop_image(imgC)
imgA = transforms.functional.to_tensor(imgA)
imgB = transforms.functional.to_tensor(imgB)
imgC = transforms.functional.to_tensor(imgC)
model = VggEncoder()
model = torch.nn.DataParallel(model).cuda()
ckpt = torch.load('data/net.pth')
model.load_state_dict(ckpt)
#print model
input_tensor = torch.cat([imgA, imgB, imgC], 0).view(1, 9, 224, 224).cuda()
start = time.time()
preds = model(input_tensor)
print(time.time() -start)
faces3d = tools.preds_to_shape(preds[0].detach().cpu().numpy())
tools.write_ply(os.path.join(options.save_dir, 'shape.ply'), faces3d[0]/1000, faces3d[1])
