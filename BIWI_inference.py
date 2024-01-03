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
df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\deep-head-pose\pose_anotations_BIWI.csv')
df['path'] = df['path'].apply(lambda x: x[1:])
data_list = df['path'].tolist()
pictures_path = r'C:\Users\isaac\PycharmProjects\Pose_images\faces_0'

# %%
dictionary = {}
for x in data_list:
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
df1 = pd.DataFrame(columns=['path', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2'])
for x in dictionary.keys():
    if len(dictionary[x]) == 1:
        df1 = df1.append({'path': x, 'bb_x1': dictionary[x][0][0], 'bb_y1': dictionary[x][0][1], 'bb_x2': dictionary[x][0][2],
                          'bb_y2': dictionary[x][0][3]}, ignore_index=True)
# %%
df1=df1.merge(df, on='path', how='left')
# %%
df1.to_csv(r'C:\Users\isaac\PycharmProjects\deep-head-pose\pose_anotations_BIWI.csv', index=False)
