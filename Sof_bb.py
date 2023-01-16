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
define_img_size(480)

# model_path = "models/pretrained/version-RFB-640.pth"
#
# net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
# net.load(model_path)
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)

# model_path = "models/pretrained/version-slim-320.pth"
model_path = "models/pretrained/version-slim-640.pth"
net = create_mb_tiny_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)


net.load(model_path)

#%%
import pandas as pd
import os
df=pd.read_csv('hpalabels1.csv')
# df['path']=df['path'].apply(lambda x: x.replace(r'C:\Users\isaac\Downloads\vlogs1\classified','C:\\Users\\isaac\\PycharmProjects\\tensorflow_filter\\classified\\'))
undesirev_col='Unnamed: 0'
df.drop(undesirev_col,axis=1,inplace=True)

paths=df['path'].tolist()
# paths=[os.path.join(r'C:\Users\isaac\Downloads\original_images',x) for x in paths ]
# os.path.join(r'C:\Users\isaac\Downloads\original_images',path)
#%%
dictionary={}
list1=[]
for x in range(len(df)):
    boxes_pic=[]
    path=df['path'][x]
    # path=os.path.join(r'C:\Users\isaac\Downloads\original_images',path)
    img=Image.open(path)
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, .75)
    if len(probs)>0:
        if probs[-1]<0.85:
            list1.append(path)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [box[0], box[1], box[2], box[3]]
        boxes_pic.append(box1)
    #     ax.add_patch(rect)
    # plt.show()
    # break
    dictionary[path]=boxes_pic


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
img=Image.open(keys_list[random_index])
img=img.convert('RGB')
img=np.array(img)
boxes, labels, probs =predictor.predict(img,1000 / 2, .75 )
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
dictionary_copy=dictionary.copy()
#%%
faces = 2
indexes = [x for x in dictionary_copy.keys() if len(dictionary_copy[x]) == faces]
print(len(indexes))
for x in indexes:

    boxes=dictionary_copy[x]
    box1=boxes[0]
    # idx=paths.index(x)
    box2 =boxes[1]
    #calculate the area of the boxes
    area1=(box1[2]-box1[0])*(box1[3]-box1[1])
    area2=(box2[2]-box2[0])*(box2[3]-box2[1])
    if area1>area2:
        if box1[0]<box2[0] and box1[1]<box2[1] and box1[2]>box2[2] and box1[3]>box2[3]:
            dictionary_copy[x]=[box1]
            indexes.remove(x)
    else:
        if box2[0]<box1[0] and box2[1]<box1[1] and box2[2]>box1[2] and box2[3]>box1[3]:
            dictionary_copy[x]=[box2]
            indexes.remove(x)

#%%
def check_intesection_of_boxes(box1,box2):
    #check if the boxes intersect and their intersection area
    x1,y1,x2,y2=box1
    x3,y3,x4,y4=box2
    if x1>x4 or x3>x2:
        return 0
    if y1>y4 or y3>y2:
        return 0
    x_overlap=max(0,min(x2,x4)-max(x1,x3))
    y_overlap=max(0,min(y2,y4)-max(y1,y3))
    intersection_area=x_overlap*y_overlap
    return intersection_area

from kutils.image.face_ops import extract_face
from kutils.image.face_ops import dlib_detection_to_sides
for x in indexes:

    boxes=dictionary[x]

    box1,box2=boxes
    img=Image.open(x)
    img=img.convert('RGB')
    img=np.array(img)
    try:
        img,dets=extract_face(img,1)
        top, right, bottom, left = dlib_detection_to_sides(dets)
        dlib_box=[left,top,right,bottom]
        intersection_area1=check_intesection_of_boxes(box1,dlib_box)
        intersection_area2=check_intesection_of_boxes(box2,dlib_box)
        if intersection_area1>intersection_area2:
            dictionary_copy[x]=[box1]
        else:
            dictionary_copy[x]=[box2]
    except:
        pass
    # plt.imshow(img)
    # plt.show()

# print(dlib_detection_to_sides(dets))
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
dictionary_copy_copy=dictionary_copy.copy()
#%%
kesy_list=list(dictionary_copy_copy.keys())
for x in kesy_list:
    boxes=dictionary_copy_copy[x]
    if len(boxes)<1:
        dictionary_copy_copy.pop(x)
#%%
number_list=[]
for x in dictionary_copy_copy.keys():
    number_list.append(len(dictionary_copy_copy[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))
#count ocurrances of each number from 1 to 6
import collections
counter=collections.Counter(number_list)
print(counter)
#%%
paths=[]
x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
x4=[]
y4=[]
x5=[]
y5=[]
lentes_claros=[]
lentes_oscuros=[]
ilumincacion=[]
cara_cubierta=[]
postura_inadecuada=[]
sombrero=[]
id=[]
bb_x1=[]
bb_y1=[]
bb_x2=[]
bb_y2=[]
for x in range(len(df)):
    if df.iloc[x][0] in dictionary_copy_copy.keys():
        boxes=dictionary_copy_copy[df.iloc[x][0]]
        box=boxes[0]
        paths.append(df.loc[x][0])
        x1.append(df.loc[x][1])
        y1.append(df.loc[x][2])
        x2.append(df.loc[x][3])
        y2.append(df.loc[x][4])
        x3.append(df.loc[x][5])
        y3.append(df.loc[x][6])
        x4.append(df.loc[x][7])
        y4.append(df.loc[x][8])
        x5.append(df.loc[x][9])
        y5.append(df.loc[x][10])
        lentes_claros.append(df.loc[x][11])
        lentes_oscuros.append(df.loc[x][12])
        ilumincacion.append(df.loc[x][13])
        cara_cubierta.append(df.loc[x][14])
        postura_inadecuada.append(df.loc[x][15])
        sombrero.append(df.loc[x][16])
        # id.append(df.loc[x][17])
        bb_x1.append(box[0])
        bb_y1.append(box[1])
        bb_x2.append(box[2])
        bb_y2.append(box[3])
#%%
print(len(paths),
len(x1),
len(y1),
len(x2),
len(y2),
len(x3),
len(y3),
len(x4),
len(y4),
len(x5),
len(y5),
len(lentes_claros),
len(lentes_oscuros),
len(ilumincacion),
len(cara_cubierta),
len(postura_inadecuada),
len(sombrero),
len(id),
len(bb_x1),
len(bb_y1),
len(bb_x2),
len(bb_y2),)
final_df=pd.DataFrame({
    'path':paths,
    'x1':x1,
    'y1':y1,
    'x2':x2,
    'y2':y2,
    'x3':x3,
    'y3':y3,
    'x4':x4,
    'y4':y4,
    'x5':x5,
    'y5':y5,
    'lentes_claros':lentes_claros,
    'lentes_oscuros':lentes_oscuros,
    'ilumincacion':ilumincacion,
    'cara_cubierta':cara_cubierta,
    'postura_inadecuada':postura_inadecuada,
    'sombrero':sombrero,
    'bb_x1':bb_x1,
    'bb_y1':bb_y1,
    'bb_x2':bb_x2,
    'bb_y2':bb_y2

})
final_df.to_csv('HPAD.csv',index=False)