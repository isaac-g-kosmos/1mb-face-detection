import random
import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
import os
from vision.ssd.config.fd_config import define_img_size
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(320)

# model_path = "models/pretrained/version-RFB-640.pth"
#
# net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
# net.load(model_path)
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)

# model_path = "models/pretrained/version-slim-320.pth"
model_path = "models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)

net.load(model_path)
#%%
paths=os.listdir(r'C:\Users\isaac\Downloads\wider_images\test\BENCHAMARK')
#%%
dictionary={}
for x in paths:

    boxes_pic=[]
    img=Image.open(os.path.join(r'C:\Users\isaac\Downloads\wider_images\test\BENCHAMARK',x))
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, .99)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        boxes_pic.append(box1)
        ax.add_patch(rect)
    # plt.show()
    # break
    dictionary[x]=boxes_pic
#%%

import json
with open('wider_face_test.json', 'w') as fp:
    json.dump(dictionary, fp)
