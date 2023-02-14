import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
#load_json
import json
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
import shutil
#%%
video_frames_path=r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images'
target_path=r'labeled_images'
# load dict from pickle
import pickle
with open(r'output_dict.pkl', 'rb') as handle:
    output_dict = pickle.load(handle)
#%%
interview_path=r'interview_images'
for x in  output_dict.keys():
    inner_dict= output_dict[x]
    img=Image.open(os.path.join(interview_path,x))
    fig,ax= plt.subplots()
    ax.imshow(img)
    try:
        if  len(inner_dict)==0:
            #save to target path
            plt.savefig(os.path.join(target_path,x))
        else:
            for y in inner_dict.keys():
                boxes= inner_dict[y]['box']
                labels= inner_dict[y]['labels']
                pose=labels[4]
                if pose>0.8:
                    ax.add_patch(plt.Rectangle((boxes[0],boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],fill=False,edgecolor='red',linewidth=2))
                else:
                    ax.add_patch(plt.Rectangle((boxes[0],boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],fill=False,edgecolor='green',linewidth=2))
            # plt.show()
            plt.savefig(os.path.join(target_path,x))
            plt.close()
    except:
        print(x)
        pass
import cv2
import numpy as np
import glob
import os
img_array = []
path1=r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images'
#%%
list1=os.listdir('labeled_images')
#order from smaller to larger
list1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
list1=[os.path.join('labeled_images',x) for x in list1]
#%%
for filename in list1:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('interview.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()