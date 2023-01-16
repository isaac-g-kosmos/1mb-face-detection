import random
import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
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
import pandas as pd
import os
pictures_path=r'C:\Users\isaac\PycharmProjects\sat-robot\stock_images'
data_list=os.listdir(pictures_path)
#%%
dictionary={}
for x in data_list:
    boxes_pic=[]
    path=os.path.join(pictures_path,x)

    img=Image.open(path)
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, .90)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [box[0], box[1], box[2], box[3]]
        boxes_pic.append(box1)
        # ax.add_patch(rect)
    # plt.show()
    # break
    dictionary[x]=boxes_pic
#%%
number_list=[]
for x in dictionary.keys():
    number_list.append(len(dictionary[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))

#count ocurrances of each number from 1 to 6
import collections
counter=collections.Counter(number_list)
print(counter)




#%%
faces=2
indexes=[x for x in range(len(number_list)) if number_list[x] ==faces]
#randomly sample 1 index
import random
random_index=random.choice(indexes)

keys_list=list(dictionary.keys())
img=Image.open(os.path.join(pictures_path,keys_list[random_index]))
img=img.convert('RGB')
img=np.array(img)
boxes, labels, probs =predictor.predict(img,1000 / 2, 0.99 )
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
dictionary_copy =dictionary.copy()
faces=2
indexes=[x for x in range(len(number_list)) if number_list[x] ==faces]
keys_list=list(dictionary.keys())

for idx in indexes:
    img=Image.open(os.path.join(pictures_path,keys_list[idx]))
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, 0.99 )
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    np_boxes = boxes.cpu().detach().numpy()
    if len(np_boxes)==1:
        box=np_boxes[0]
        dictionary_copy[keys_list[idx]]=[[box[0], box[1], box[2], box[3]]]
    elif len(np_boxes)<1:
        dictionary_copy[keys_list[idx]] =[]
number_list=[]
for x in dictionary_copy.keys():
    number_list.append(len(dictionary_copy[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))

#count ocurrances of each number from 1 to 6
import collections
counter=collections.Counter(number_list)
print(counter)
#%%
faces=2
indexes=[x for x in range(len(number_list)) if number_list[x] ==faces]
#randomly sample 1 index
import random
random_index=indexes[2]

keys_list=list(dictionary_copy.keys())
img=Image.open(os.path.join(pictures_path,keys_list[random_index]))
img=img.convert('RGB')
img=np.array(img)
boxes, labels, probs =predictor.predict(img,1000 / 2, 0.99 )
fig, ax = plt.subplots(1)
ax.imshow(img)
np_boxes = boxes.cpu().detach().numpy()
cout=1
for box in np_boxes:
    rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
    anotation=plt.annotate(cout, (box[0], box[1]), color='red')
    cout+=1
    ax.add_patch(rect)
plt.show()
print(probs)
print(keys_list[random_index])
#%%
indexes.remove(indexes[2])
#%%
for idx in indexes:
    print([dictionary_copy[keys_list[idx]][0]])
    # break
    dictionary_copy[keys_list[idx]]=[dictionary[keys_list[idx]][0]]
#%%
number_list=[]
for x in dictionary_copy.keys():
    number_list.append(len(dictionary_copy[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))

#count ocurrances of each number from 1 to 6
import collections
counter=collections.Counter(number_list)
print(counter)
#%%
keys_list=list(dictionary_copy.keys())
faces=0
indexes=[x for x in range(len(number_list)) if number_list[x] ==faces]
for idx in indexes:
    try:
        img=Image.open(os.path.join(pictures_path,keys_list[idx]))
        img=img.convert('RGB')
        img=np.array(img)
        boxes, labels, probs =predictor.predict(img,1000 / 2, 0.5 )
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        np_boxes = boxes.cpu().detach().numpy()
        if len(np_boxes)>0:
            box=np_boxes[0]
            dictionary_copy[keys_list[idx]]=[[box[0], box[1], box[2], box[3]]]
    except:
        print('error')
        pass
number_list=[]
for x in dictionary_copy.keys():
    number_list.append(len(dictionary_copy[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))
import collections
counter=collections.Counter(number_list)
print(counter)
#%%
paths=[]
bb_x1=[]
bb_y1=[]
bb_x2=[]
bb_y2=[]
for x in dictionary_copy.keys():
    if len(dictionary_copy[x])>0:
        paths.append(x)
        bb_x1.append(dictionary_copy[x][0][0])
        bb_y1.append(dictionary_copy[x][0][1])
        bb_x2.append(dictionary_copy[x][0][2])
        bb_y2.append(dictionary_copy[x][0][3])
#%%
import pandas as pd
df=pd.DataFrame({'path':paths,'bb_x1':bb_x1,'bb_y1':bb_y1,'bb_x2':bb_x2,'bb_y2':bb_y2})
df.to_csv('stock_images.csv',index=False)
#%%
import pandas as pd
df=pd.read_csv('stock_images.csv')
#%%