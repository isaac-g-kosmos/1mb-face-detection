import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#%%
df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\final_df2.csv')
df.drop(columns=['id'], inplace=True)
# df['path']=df['path'].apply(lambda x: x.replace(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\classified',
#                                                 '/home/ubuntu/vlogs/').replace('\\','/'))
#%%
df2=pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\new_augmentations_18-10.csv')
# df2['path']=df2['path'].apply(lambda x: x.replace(r"C:\Users\isaac\PycharmProjects\tensorflow_filter\dataset_pictures",
#                                                 '/home/ubuntu/dataset_pictures').replace('\\','/'))

# #%%
df1 = pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\SoF_dataset.csv')
df1['path'] = df1['path'].apply(lambda x: os.path.join(r'C:\Users\isaac\Downloads\original_images',x))
#%%
df3=pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\Dark_glasses_landmarks2.csv')
df3['path']= df3['path'].apply(lambda x: x.replace(r'/home/ubuntu/Dark_glasses/',''))

df3['path']= df3['path'].apply(lambda x: os.path.join(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\dark_glasses', x))
#%%

df4 = pd.read_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\meglasses2.csv')
df4['path'] = df4['path'].apply(lambda x: os.path.join(r'C:\Users\isaac\Downloads\MeGlass\data\MeGlass_ori', x))
# %%
df=pd.concat([df,df1,df2,df3,df4], ignore_index=True)
import uuid
dark_glasses_only=df[df['lentes_oscuros']==1]
dark_glasses_only.reset_index(inplace=True)
for _ in range(3):
    for x in range(len(dark_glasses_only)):
        try:
            name=str(uuid.uuid4())+'.jpg'
            # print(name)
            path=dark_glasses_only['path'].loc[x]
            img=Image.open(path)
            img=img.convert('RGB')
            #randomly rotate between -45 and 45 degrees
            angle=np.random.randint(-45,45)
            img=img.rotate(angle)
            #randomly flip
            # flip=np.random.randint(0,2)
            # if flip==1:
            #     img=img.transpose(Image.FLIP_LEFT_RIGHT)
            #save the image to C:\Users\isaac\PycharmProjects\tensorflow_filter\albaugmentations
            img.save(os.path.join(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\albaugmentations',name))
            # print('saved')
        except Exception as e:
            print(e)
            # print('error')
            pass
