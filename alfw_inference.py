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
with open(r'C:\Users\isaac\PycharmProjects\deep-head-pose\original_df.pkl', 'rb') as handle:
    df = pickle.load(handle)

data_list = df['original_pictures'].tolist()
pictures_path = r'C:\Users\isaac\PycharmProjects\Pose_images\300W_LP'
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

# count ocurrances of each number from 1 to 6
import collections

counter = collections.Counter(number_list)
print(counter)

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
# %%
import pickle

with open(r'C:\Users\isaac\PycharmProjects\deep-head-pose\300W_LP.pkl', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
#unpickle
with open(r'C:\Users\isaac\PycharmProjects\deep-head-pose\300W_LP.pkl', 'rb') as handle:
    dictionary = pickle.load(handle)
# %%
dictionary_copy = dictionary.copy()


def iou(bb1, bb2):
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = bb2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
def ioB(bb1, original_box):
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = original_box
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / float( boxBArea )
    return iou


# parse thouygh the boundingboxe  and keep the ones with the large iou with the roi in the dataframe
coount = 0
for x in dictionary_copy.keys():
    coount += 1
    print(coount)
    landmarks = df[df['original_pictures'] == x]['landmarks'].values[0]
    x_min = np.min(landmarks[0, :])
    y_min = np.min(landmarks[1, :])
    x_max = np.max(landmarks[0, :])
    y_max = np.max(landmarks[1, :])
    orginal_bb = [x_min, y_min, x_max, y_max]
    predicted_boxes = dictionary_copy[x]
    bb_max = -1000
    for box in predicted_boxes:
        iou_value = ioB(box, orginal_bb)
        if iou_value > bb_max:
            bb_max = iou_value
            max_box = box
            dictionary_copy[x] = [max_box, bb_max, orginal_bb,landmarks]
    if bb_max <= 0:
        dictionary_copy[x] = []
# %%
# Create a df with the  dictionary copy data
new_df = pd.DataFrame(columns=['path', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'iou','original_BB'])
for x in dictionary_copy.keys():
    if len(dictionary_copy[x]) > 0:
        new_df = new_df.append({'path': x, 'bb_x1': dictionary_copy[x][0][0], 'bb_y1': dictionary_copy[x][0][1],
                                'bb_x2': dictionary_copy[x][0][2], 'bb_y2': dictionary_copy[x][0][3],
                                'iou': dictionary_copy[x][1], 'original_BB':dictionary_copy[x][2],
                                "landmarks":dictionary_copy[x][3]}, ignore_index = True)
#%%
#check weather  how  many landmarks are inside the bounding
def check_inside(bb,landmarks):
    x_min = bb[0]
    y_min = bb[1]
    x_max = bb[2]
    y_max = bb[3]
    bool_x = np.logical_and(landmarks[0, :] >= x_min, landmarks[0, :] <= x_max)
    bool_y = np.logical_and(landmarks[1, :] >= y_min, landmarks[1, :] <= y_max)
    bool_xy = np.logical_and(bool_x, bool_y)
    return np.sum(bool_xy)

new_df['inside_landmarks']=new_df.apply(lambda x: check_inside(x['bb_x1':'bb_y2'].values,x['landmarks']),axis=1)
#%%
#pickle  new df
import pickle
with open(r'C:\Users\isaac\PycharmProjects\deep-head-pose\new_df.pkl', 'wb') as handle:
    pickle.dump(new_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# unplicle
with open(r'C:\Users\isaac\PycharmProjects\deep-head-pose\new_df.pkl', 'rb') as handle:
    new_df = pickle.load(handle)

# %%
# new_df['iou'].hist(bins=20)
# plt.show()
new_df['iou'].hist(bins=35)
plt.show()
pictures_path = r'C:\Users\isaac\PycharmProjects\Pose_images\300W_LP'
# %%
def augmented_cut(BB, width, height, leeway=0):
    BB[:, 0] = BB[:, 0] - leeway * width / 2
    BB[:, 1] = BB[:, 1] - leeway * height / 2
    BB[:, 2] = BB[:, 2] + leeway * width
    BB[:, 3] = BB[:, 3] + leeway * height

    return BB


# samples_df=new_df[new_df['iou']<=0.45]
samples_df = new_df[new_df['iou'] >= 0.4]
# samples_df = new_df[new_df['inside_landmarks'] >= 45]
# samples_df = new_df[new_df['inside_landmarks'] <= 50]
print(len(samples_df))
# smaple and grapth 10 images with their BB
# box = samples_df[['bb_x1', 'bb_y1', 'bb_x2', 'bb_y2']].values
# box_x_lenght = box[:, 2] - box[:, 0]
# box_y_height = box[:, 3] - box[:, 1]
# box = augmented_cut(box, box_x_lenght, box_y_height, .2)
#
# box[box < 0] = 0
# samples_df[['bb_x1', 'bb_y1', 'bb_x2', 'bb_y2']] = box
#%%
sample_count=0
#%%

samples = samples_df.sample(10)
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
coount = 0
for index, row in samples.iterrows():
    img = Image.open(os.path.join(pictures_path, row['path']))
    img = img.convert('RGB')
    img = np.array(img)
    ax[coount % 2, int(coount / 2)].imshow(img)

    ax[coount % 2, int(coount / 2)].add_patch(
        plt.Rectangle((row['bb_x1'], row['bb_y1']), row['bb_x2'] - row['bb_x1'], row['bb_y2'] - row['bb_y1'],
                      fill=False, color='red'))
    ax[coount % 2, int(coount / 2)].add_patch(
        plt.Rectangle((row['original_BB'][0], row['original_BB'][1]), row['original_BB'][2] - row['original_BB'][0],
                      row['original_BB'][3] - row['original_BB'][1],
                      fill=False, color='Blue'))
    #plot landdmarks
    # ax[coount % 2, int(coount / 2)].scatter(row['landmarks'][0, :], row['landmarks'][1, :], s=10, marker='.', c='w')
    #write number of landmarks inside BB
    # ax[coount % 2, int(coount / 2)].text(0, 0, np.round(row['inside_landmarks'],2), color='black')
    #write iou
    ax[coount % 2, int(coount / 2)].text(300, 0, np.round(row['iou'],2), color='black')
    coount += 1
    sample_count+=1
plt.show()

# %%
#add the pose row from df to the new df
#%%
# renamane orignial_pictures to path in df
df.rename(columns={'original_pictures': 'path'}, inplace=True)
#%%
#concat by path with new _df
new_df=new_df.merge(df, on='path', how='left')
#%%
new_df=new_df[['path','bb_x1', 'bb_y1', 'bb_x2', 'bb_y2','pose']]
new_df['pitch']=new_df['pose'].apply(lambda x: x[0])
new_df['yaw']=new_df['pose'].apply(lambda x: x[1])
new_df['roll']=new_df['pose'].apply(lambda x: x[2])
#%%
# drop pose
new_df.drop(columns=['pose'],inplace=True)
#%%
new_df.to_csv(r'C:\Users\isaac\PycharmProjects\deep-head-pose\300W_1.csv',index=False)
