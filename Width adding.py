import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
df=pd.read_csv(r'vgg_face_hat.csv')
#%%
heights=[]
widths=[]
for x in range(len(df)):
    path=df['path'][x]
    img=Image.open(os.path.join(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\vgg_face',path))
    width, height = img.size
    img=np.array(img)
    heights.append(height)
    widths.append(width)
    # plt.imshow(img)
    # plt.show()
#%%
df['width']=widths
df['height']=heights
#%%M
df.to_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\hat_df\vgg_face_hat.csv',index=False)