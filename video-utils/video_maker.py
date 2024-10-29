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