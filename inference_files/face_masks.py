import random
import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
from vision.ssd.config.fd_config import define_img_size
import cv2
import pandas as pd
import os
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(320)


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
#%%

df=pd.read_csv(r'C:\Users\isaac\PycharmProjects\COFW\kagle_masks.csv')


pictures_path=r'C:\Users\isaac\PycharmProjects\COFW\FM\images'

file_series=df['filename']
file_series=file_series.apply(lambda x: os.path.basename(x))
#group by filename and count
file_series=file_series.groupby(file_series).count()
file_series.hist(bins=100)
plt.show()
file_series=file_series[file_series<10]

df=df[df['filename'].apply(lambda x: os.path.basename(x)).isin(file_series.index)]
#%%
data_list=df['filename'].tolist()
data_list=[os.path.basename(x) for x in data_list]
#%%
dictionary={}
for x in data_list:
    boxes_pic=[]
    path=os.path.join(pictures_path,x)

    img=Image.open(path)
    img=img.convert('RGB')
    img=np.array(img)
    boxes, labels, probs =predictor.predict(img,1000 / 2, .85)
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
#%%
def iou(bb1,bb2):
    x11,y11,x12,y12=bb1
    x21,y21,x22,y22=bb2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
#%%
label_dict={0:'mask_weared_incorrect',
1:'without_mask',
2:'with_mask'}
dictionary2={}
for image in dictionary.keys():
    boxes=dictionary[image]
    dictionary2[image]=[]
    small_df=df[df['filename']==image]

    for box in boxes:
        correct=0
        x1,y1,x2,y2=box
        for row in small_df.iterrows():
            x1_,y1_,x2_,y2_=row[1]['min_x'],row[1]['min_y'],row[1]['max_x'],row[1]['max_y']
            label=np.argmax(row[1][['mask_weared_incorrect','without_mask','with_mask']])
            label=label_dict[label]
            iou_=iou(box,[x1_,y1_,x2_,y2_])
            if iou_>0.3:
                correct=1
                dictionary2[image].append({'box':box,'label':label})
                break
            else:
                correct=0



#%%
number_list=[]
for x in dictionary2.keys():
    number_list.append(len(dictionary2[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))
#%%
#wtite dictionary2 into a dataframe to csv
df2=pd.DataFrame(columns=['filename','box','label'])
for image in dictionary2.keys():
    for box in dictionary2[image]:
        df2=df2.append({'filename':image,'box':box['box'],'label':box['label']},ignore_index=True)
df2.to_csv('kagle_masks2.csv',index=False)
#%%
#pickle df2
import pickle
with open('kagle_masks2.pickle', 'wb') as handle:
    pickle.dump(df2, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
files=set(df2['filename'].tolist())
#%%
sample=files.pop()
sample_df=df2[df2['filename']==sample]
#graph all the boxes in the images
img=Image.open(os.path.join(pictures_path,sample))
img=img.convert('RGB')
img=np.array(img)
fig, ax = plt.subplots(1)
ax.imshow(img)
for row in sample_df.iterrows():
    box=row[1]['box']
    text=row[1]['label']
    rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
    ax.text(box[0], box[1], text, fontsize=10, color='red')

    ax.add_patch(rect)
plt.show()
#%%
# 'mask_weared_incorrect'
# 'without_mask'
# 'with_mask'
masked_images=df[df['with_mask']==1]
sample=masked_images.sample(1)
sample=sample['filename'].tolist()[0]
sample=os.path.basename(sample)
img=Image.open(os.path.join(pictures_path,sample))
img=img.convert('RGB')
img=np.array(img)
box=dictionary2[sample][0]['box']
fig, ax = plt.subplots(1)
ax.imshow(img)
rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
ax.add_patch(rect)
plt.show()

