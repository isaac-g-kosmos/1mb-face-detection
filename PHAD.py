import math

import pandas as pd
import os
good_pose_file=os.listdir(r'C:\Users\isaac\Downloads\kara2018_headposeannotations\Good_pose')
#%%
df=pd.read_csv(r"C:\Users\isaac\Downloads\kara2018_headposeannotations\headpose_groundtruth.csv",
               sep=';')
#%%
df_anotated=pd.read_csv(r"C:\Users\isaac\Downloads\kara2018_headposeannotations\headpose_annotations.csv",
                        sep=';')
all_pic_paths=r'C:\Users\isaac\Downloads\kara2018_headposeannotations\all_pics'
list_all_pics=os.listdir(all_pic_paths)
#%%
df_anotated=df_anotated[df_anotated['Tilt']==4]
df_anotated=df_anotated[df_anotated['Pan']==4]
#%%
paths=df_anotated['SampleID'].tolist()
paths=[os.path.join(all_pic_paths,x) for x in paths]
new_list=paths.copy()
for x in set(paths):
    if paths.count(x)==1:
        new_list.remove(x)
#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
rand_int=random.randint(0,len(new_list))
img=Image.open(new_list[rand_int])
img=np.array(img)
plt.imshow(img)
# plt.show()
#%%
specific_path=r'project-28-at-2022-10-18-17-41-0ee974d2.csv'
# df=pd.read_csv(specific_path)
#%%
import json
# path=r'C:\Users\isaac\PycharmProjects\tensorflow_filter\all_pics'
# files = os.listdir(path)
# files = [os.path.join(path,x) for x in files]
# %%
def label_studio_path_transform(image_path,output_path):
    image_name=image_path.split('-person')[-1]
    image_name='person'+image_name
    person_names=image_name[:11].split('.')[-1]
    matching = [s for s in list_all_pics if person_names in s]
    return os.path.join(output_path,matching[0])
def label_studio_path_transform_test(strings:str):
    if "choices" in strings:
        return json.loads(strings)['choices']
    else:
        return strings




df = pd.read_csv(specific_path)
#%%
df['image']=df['image'].apply(lambda x: label_studio_path_transform(x,r'C:\Users\isaac\Downloads\kara2018_headposeannotations\all_pics'))
df1 = pd.DataFrame(columns=['path','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5',
                            'lentes_claros',
                            'lentes_oscuros',
                            'iluminacion_indadecuada',
                            'cara_cubierta',
                            'postura_inadecuada',
                            'sombrero','id'])

# df.dropna(inplace=True)
# keep in files if they exist
# files=[x for x in files if os.path.exists(x)]
# %%
import dlib
import math
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    r'C:\Users\isaac\PycharmProjects\tensorflow_filter\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat')
# %%
lentes_claros = []
lentes_oscuros = []
iluminacion_indadecuada = []
cara_cubierta = []
postura_inadecuada = []
sombrero = []
ninguno=[]
ids=[]
def append_labels(labels):
    if type(labels) == str:
        if labels == "Lentes claros":
            lentes_claros.append(1)
        else:
            lentes_claros.append(0)
        if labels == "Uso de Lentes oscuros":
            lentes_oscuros.append(1)
        else:
            lentes_oscuros.append(0)
        if labels == "Iluminacion irregular en la cara":
            iluminacion_indadecuada.append(1)
        else:
            iluminacion_indadecuada.append(0)
        if ((labels == "Objetos cubriedno la cara")
                or (labels == "Cabello cubriendo la cara")):
            cara_cubierta.append(1)
        else:
            cara_cubierta.append(0)
        if labels == "Postura inadecuada":
            postura_inadecuada.append(1)
        else:
            postura_inadecuada.append(0)
        if labels == "Uso de un ornamento en la cabeza":
            sombrero.append(1)
        else:
            sombrero.append(0)
    elif math.isnan(labels) :
        lentes_claros.append(0)
        lentes_oscuros.append(0)
        iluminacion_indadecuada.append(0)
        cara_cubierta.append(0)
        postura_inadecuada.append(0)
        sombrero.append(0)
    else:
        if "Uso de un ornamento en la cabeza" in labels:
            sombrero.append(1)
        if "Postura inadecuada" in labels:
            postura_inadecuada.append(1)
        else:
            postura_inadecuada.append(0)
        if ("Cabello cubriendo la cara" in labels) or ("Objetos cubriedno la cara" in labels):
            cara_cubierta.append(1)
        else:
            cara_cubierta.append(0)
        if "Iluminacion irregular en la cara" in labels:
            iluminacion_indadecuada.append(1)
        else:
            iluminacion_indadecuada.append(0)
        if "Uso de Lentes oscuros" in labels:
            lentes_oscuros.append(1)
        else:
            lentes_oscuros.append(0)
        if "Lentes claros" in labels:
            lentes_claros.append(1)
        else:
            lentes_claros.append(0)
df1 = pd.DataFrame(columns=['path','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','lentes_claros',
                            'lentes_oscuros',
                            'iluminacion_indadecuada',
                            'cara_cubierta',
                            'postura_inadecuada',
                            'sombrero','id'])
#%%
paths=[]
for idx in range(0,len(df)):
    path = df.iloc[idx]['image']
    paths.append(path)
    id=df.iloc[idx]['id']
    ids.append(id)
    print(idx)
    print(id)
    labels = df.iloc[idx]['choice']
    print(labels)
    append_labels(labels)
#%%
# df1['path']=paths
# df1['lentes_claros']=lentes_claros
# df1['lentes_oscuros']=lentes_oscuros
# df1['iluminacion_indadecuada']=iluminacion_indadecuada
# df1['cara_cubierta']=cara_cubierta
# df1['postura_inadecuada']=postura_inadecuada
# df1['sombrero']=sombrero
# df1['id']=ids

np.sum(lentes_claros)
# df1.to_csv(r'hpalabels.csv',index=False)
# %%
for idx in range(0,len(df)):
    path = df.iloc[idx]['image']
    id=df.iloc[idx]['id']
    print(idx)
    # print(id)
    labels = df.iloc[idx]['choice']
    append_labels(labels)
    try:
        img = Image.open(path)
        img = np.array(img)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = predictor(img, d)
        eyes_landmarks = [(36, 39), (42, 45)]
        other_landmarks = [34, 48, 54]

        mean_landmarks = []
        for (i, j) in eyes_landmarks:
            mean_x = (shape.part(i).x + shape.part(j).x) / 2
            mean_y = (shape.part(i).y + shape.part(j).y) / 2
            mean_landmarks.append((mean_x, mean_y))
        for i in other_landmarks:
            mean_landmarks.append((shape.part(i).x, shape.part(i).y))
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        output=[]
        output.append(path)
        for landmark in mean_landmarks:
            # ax.scatter(landmark[0], landmark[1], 10, marker='x', color='blue', linewidth=5
            #            )
            output.append(landmark[0])
            output.append(landmark[1])

        labels_list=[lentes_claros[-1],
                        lentes_oscuros[-1],
                        iluminacion_indadecuada[-1],
                        cara_cubierta[-1],
                        postura_inadecuada[-1],
                        sombrero[-1]]
        output.extend(labels_list)
        output.append(id)
        df1.loc[idx] =  output
        df1.to_csv('hpalabels1.csv')
    except Exception as e:
        print(e)
        print('error')
# wandb.log({'Dataset_table': table})
# %%
for x in range(0,len(df1)):
    name= df1.iloc[x]['path']
    name=name.split('\\')[-1]
    path_check=[x.split('\\')[-1] for x in new_list]
    if name in path_check:
        df1['postura_inadecuada'][x]=0
    else:
        df1['postura_inadecuada'][x]=1
df1.to_csv('hpalabels1.csv')
# %%
np.sum(df1['postura_inadecuada'])
# %%
# fif=plt.figure()

# wandb.init(project="tensorflow_filter")
# wandb.log({'picture':fig})
# wandb.finish()
