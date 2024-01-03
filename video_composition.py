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
from math import cos, sin


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 =- size * (cos(yaw) * cos(roll)) + tdx
    y1 = -size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
video_frames_path=r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\vlog_images'
target_path=r'blogs_imagevid'
# load dict from pickle
#%%
import pickle
with open(r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\output_dict.pkl', 'rb') as handle:
    output_dict = pickle.load(handle)
#%%
import sys
interview_path=r'interview_images'
for x in  output_dict.keys():
    inner_dict= output_dict[x]
    img=Image.open(os.path.join(interview_path,x))

    try:
        if  len(inner_dict)==0:
            #save to target path
            plt.savefig(os.path.join(target_path,x))
        else:
            for y in inner_dict.keys():
                if inner_dict[y]==[]:
                  pass
                else:
                    boxes= inner_dict[y]['box']
                    labels= inner_dict[y]['label']
                    img=draw_axis(np.array(img),labels[0],labels[1],labels[2],(boxes[2]+boxes[0])/2,(boxes[3]+boxes[1])/2)
                    #save to target path with cv2 as rgb

                    cv2.imwrite(os.path.join(target_path,x),cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #print line where error occured
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        print(x)
        pass
import cv2
import numpy as np
import glob
import os
img_array = []
#%%
list1=os.listdir('blogs_imagevid')
#order from smaller to larger
list1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
list1=[os.path.join('blogs_imagevid',x) for x in list1]
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