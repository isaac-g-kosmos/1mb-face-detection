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
data_list= os.listdir(r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebA-HQ-img')
pictures_path = r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebA-HQ-img'


# %%
dictionary = {}
for x in data_list:
    boxes_pic = []
    path = os.path.join(pictures_path, x)

    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)
    boxes, labels, probs = predictor.predict(img, 1000 / 2, .85)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    # plt.show()
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [box[0], box[1], box[2], box[3]]
        boxes_pic.append(box1)
        # ax.add_patch(rect)
    # plt.show()
    # break
    dictionary[x] = boxes_pic
# %%
number_list = []
for x in dictionary.keys():
    number_list.append(len(dictionary[x]))

print(max(number_list))
print(min(number_list))
print(np.mean(number_list))

# make a df with images with only one pic their path box and label
# %%
faces = 2
indexes = [x for x in range(len(number_list)) if number_list[x] == faces]
# randomly sample 1 index
import random

random_index = random.choice(indexes)

keys_list = list(dictionary.keys())
img = Image.open(os.path.join(pictures_path, keys_list[random_index]))
img = img.convert('RGB')
img = np.array(img)
boxes, labels, probs = predictor.predict(img, 1000 / 2, 0.85)
fig, ax = plt.subplots(1)
ax.imshow(img)
np_boxes = boxes.cpu().detach().numpy()
for box in np_boxes:
    rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
    ax.add_patch(rect)
plt.show()
print(probs)
print(keys_list[random_index])
#%%
# make a df with images with only one pic their path box and label only if they excactly one bb
# %%
paths   = []
bb_x1= []
bb_y1= []
bb_x2= []
bb_y2= []

for x in dictionary.keys():
    if len(dictionary[x]) == 1:
        paths.append(x)
        bb_x1.append(dictionary[x][0][0])
        bb_y1.append(dictionary[x][0][1])
        bb_x2.append(dictionary[x][0][2])
        bb_y2.append(dictionary[x][0][3])

# %%
df=pd.DataFrame({'path':paths,'bb_x1':bb_x1,'bb_y1':bb_y1,'bb_x2':bb_x2,'bb_y2':bb_y2})
# %%
df.to_csv('CelebAMask-HQ.csv')
