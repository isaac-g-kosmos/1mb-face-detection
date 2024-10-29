import random
import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
from vision.ssd.config.fd_config import define_img_size
import cv2
import torch
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(320)
leeway = .2

# model_path = "models/pretrained/version-RFB-640.pth"
#
# net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
# net.load(model_path)                              weq
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)

# model_path = "models/pretrained/version-slim-320.pth"
model_path = "models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)

net.load(model_path)
# %%
# unpickle

pictures_path = r'C:\Users\isaac\PycharmProjects\CC_VIDS_pictures'
data_list = os.listdir(pictures_path)

dictionary = {}
counter=0
for x in data_list:
    counter+=1
    if counter%100==0:
        print(counter)
    boxes_pic = []
    path = os.path.join(pictures_path, x)

    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)
    boxes, labels, probs = predictor.predict(img, 1000 / 2, .9)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    # plt.show()
    np_boxes = boxes.cpu().detach().numpy()

    if len(np_boxes)==1:
        for box in np_boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
            box1 = [box[0], box[1], box[2], box[3]]
            dictionary[x]=box1
#%%
empty_new={}
for key in dictionary.keys():
    empty_new[key]=[int(x) for x in dictionary[key]]
#%%
import json

with open("bb_resulsts.json", 'w') as json_file:
    json.dump(empty_new, json_file)