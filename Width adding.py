import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
df=pd.read_csv(r'scrapped_hats2.csv')
#%%
heights=[]
widths=[]
for x in range(len(df)):
    path=df['path'][x]
    img=Image.open(os.path.join(r'C:\Users\isaac\PycharmProjects\sat-robot\burka',path))
    width, height = img.size
    img=np.array(img)
    heights.append(height)
    widths.append(width)
    # plt.imshow(img)
    # plt.show()
#%%
df['width']=widths
df['height']=heights
df['sombrero']=1
# #%%
# df['sombrero'][df['path'].str.startswith('hats')]=1
# #%%
# df['sombrero'].value_counts()
#%%
df.drop(columns=['Unnamed: 0'],inplace=True)
#%%M
# df.to_csv(r'C:\Users\isaac\PycharmProjects\insightface\scrapped_hats2.csv',index=False)
df.to_csv(r'scrapped_hats2.csv',index=False)


# train_occlusion_accuracy 0.97
# train_occlusion_f1 0.97
# train_occlusion_presicion 0.96
# train_occlusion_recall 0.98
#
# test_occlusion_accuracy 0.42
# test_occlusion_acum_loss 19.66
# test_occlusion_presicion 0.42
# test_occlusion_recall 1
# test_occlusion_f1 0.59
#
# val_occlusion_accuracy 0.96
# val_occlusion_f1 0.84
# val_occlusion_presicion 0.79
# val_occlusion_recall 0.91


