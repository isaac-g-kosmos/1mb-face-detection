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
model_path = "models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)


net.load(model_path)

#%%
import pandas as pd
import os
df=pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\AFLW_df.csv')
# df['path']=df['path'].apply(lambda x: x.replace(r'C:\Users\isaac\Downloads\vlogs1\classified','C:\\Users\\isaac\\PycharmProjects\\tensorflow_filter\\classified\\'))
# undesirev_col='Unnamed: 0'
# df.drop(undesirev_col,axis=1,inplace=True)

# paths=df['path'].tolist()
# paths=[os.path.join(r'C:\Users\isaac\Downloads\original_images',x) for x in paths ]
# # os.path.join(r'C:\Users\isaac\Downloads\original_images',path)
#%%
dictionary={}
for x in range(len(df)):
    boxes_pic=[]
    path=df['path'][x]
    path=os.path.join(r'C:\Users\isaac\Downloads\original_images',path)
    img=Image.open(path)
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, .95)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [box[0], box[1], box[2], box[3]]
        boxes_pic.append(box1)
        ax.add_patch(rect)
    plt.show()
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
import re
idx=150
path = df['path'][idx]
path = os.path.join(r'C:\Users\isaac\Downloads\original_images', path)
bb_box=df['faceRect'][idx]
bb_box=re.findall("\d+\.\d+",bb_box)

print(bb_box)

bb_box=[float(x) for x in bb_box]
img=Image.open(path)
img=img.convert('RGB')
img=np.array(img)
fig, ax = plt.subplots(1)
slice=img[int(bb_box[1]):int(bb_box[1]+bb_box[3]),int(bb_box[0]):int(bb_box[0]+bb_box[2])]
#%%
dictionary_copy=dictionary.copy()
#%%
faces = 2
indexes = [x for x in dictionary_copy.keys() if len(dictionary_copy[x]) == faces]
print(len(indexes))
for x in indexes:

    boxes=dictionary_copy[x]
    box1=boxes[0]
    box2=boxes[1]
    # box2 =df.iloc[idx]['faceRect']
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
#%%
indexes = [x for x in dictionary_copy.keys() if len(dictionary_copy[x]) != 1]
for x in indexes:
    bb_box=df.iloc[x]['faceRect']
    print(x)
    print(df.iloc[x]['path'])
    bb_box = re.findall("\d+\.\d+", bb_box)
    bb_box = [float(x) for x in bb_box]
    # bb_box=[bb_box[0],bb_box[1],bb_box[0]+bb_box[2],bb_box[1]+bb_box[3]]
    #
    bb_box = [bb_box[0], bb_box[1], bb_box[0] + bb_box[3], bb_box[1] + bb_box[2]]
    dictionary_copy[x]=[bb_box]
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
heights=[]
widths=[]
for x in range(len(df)):

    boxes=dictionary_copy[x]
    box=boxes[0]
    paths.append(df['path'][x])
    x1.append(df['x1'][x])
    y1.append(df['y1'][x])
    x2.append(df['x2'][x])
    y2.append(df['y2'][x])
    x3.append(df['x3'][x])
    y3.append(df['y3'][x])
    x4.append(df['x4'][x])
    y4.append(df['y4'][x])
    x5.append(df['x5'][x])
    y5.append(df['y5'][x])
    lentes_claros.append(df['lentes_claros'][x])
    lentes_oscuros.append(df['lentes_oscuros'][x])
    ilumincacion.append(df['iluminacion_indadecuada'][x])
    cara_cubierta.append(df['cara_cubierta'][x])
    postura_inadecuada.append(df['postura_inadecuada'][x])
    sombrero.append(df.loc[x]['sombrero'])
    bb_x1.append(box[0])
    bb_y1.append(box[1])
    bb_x2.append(box[2])
    bb_y2.append(box[3])
    heights.append(df.loc[x]['height'])
    widths.append(df.loc[x]['width'])
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
    'bb_y2':bb_y2,
    'width': widths,
    'height': heights
})
#drop duplicates
final_df=final_df.drop_duplicates(subset=['path'])
#%%
final_df.to_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\SoF_dataset.csv',index=False)