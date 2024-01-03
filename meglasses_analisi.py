import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#%%
frame_path=r'C:\Users\isaac\PycharmProjects\tensorflow_filter\dark_glasses_data\meglasses2_implied_BB.pkl'
df=pd.read_pickle(frame_path)
df_meglasses=pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\dark_glasses_data\meglasses2.csv')
#%%
df=df.merge(df_meglasses,on='path')
#%%
picture_path=r'C:\Users\isaac\Downloads\MeGlass\data\MeGlass_ori'
sample=df.iloc[100]
bb=sample['implied_BB']
img=cv2.imread(os.path.join(picture_path,sample['path']))
# img_cut=img[bb[1]:bb[3],bb[0]:bb[2]]
# plt.annotate('bb',xy=(bb[0],bb[1]),xytext=(bb[0],bb[1]),arrowprops=dict(facecolor='red', shrink=0.05))
# plt.annotate('bb',xy=(bb[2],bb[3]),xytext=(bb[2],bb[3]),arrowprops=dict(facecolor='red', shrink=0.05))
img2=cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),255,2)
plt.imshow(img2)
plt.show()

actual_bb=sample[['bb_x1','bb_y1','bb_x2','bb_y2']].values
img=cv2.imread(os.path.join(picture_path,sample['path']))
img_cut=img[int(actual_bb[1]):int(actual_bb[3]),int(actual_bb[0]):int(actual_bb[2])]
plt.imshow(img_cut)
# plt.show()

print(bb[1],bb[3],bb[0],bb[2])
print(int(actual_bb[1]),int(actual_bb[3]),int(actual_bb[0]),int(actual_bb[2]))