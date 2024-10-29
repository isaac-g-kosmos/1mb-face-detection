import os
import cv2
import tensorflow as tf
from pre_process import augmented_cut
import random
import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

from vision.ssd.config.fd_config import define_img_size
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(1280)

model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)
leeway = 0.2

net.load(model_path)
#%%
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.applications import MobileNetV3Small
inputs = Input(shape=(256, 256, 3), name='main_input')#1,8,8,576
movnet = MobileNetV3Small(include_top=False)(inputs)
movnet.trainable = False
pose_stem=Conv2D(480, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(movnet)
pose_stem=Activation("LeakyReLU")(pose_stem)
pose_stem=Conv2D(320,kernel_size =(1, 1), strides=(1,1), padding='same', activation='relu')(pose_stem)
pose_stem=Activation("LeakyReLU")(pose_stem)
pose_stem=Conv2D(144, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(pose_stem)
pose_stem=Activation("LeakyReLU")(pose_stem)



#8x8x20=1280
pose_stem=Flatten()(pose_stem)
pose_stem=Dense(4096, activation='relu')(pose_stem)
pose_stem=Dropout(.1)(pose_stem)
pose_stem=Dense(4096, activation='relu')(pose_stem)
pose_stem=Dropout(.1)(pose_stem)
pose_stem=Dense(66, activation='relu')(pose_stem)
pose_stem=Dropout(.1)(pose_stem)
yaw=Dense(66, activation='relu')(pose_stem)
pitch=Dense(66, activation='relu')(pose_stem)
roll=Dense(66, activation='relu')(pose_stem)
model=Model(inputs=inputs, outputs=[yaw,pitch,roll])
#%%
model.load_weights(
    r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\movnet_scarlet-shape-73-24.h5")
# %%
img_path = r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\vlog_images'

images= os.listdir(img_path)
#%%
output_dict= {}
#%%

idx_tensor = [idx for idx in range(66)]
idx_tensor = tf.convert_to_tensor(idx_tensor)
idx_tensor = tf.cast(idx_tensor, tf.float32)
for image in images[:5000]:
    try:
        output_dict[image] = {}
        img=Image.open(os.path.join(img_path,image))
        #down  sample to 500X500
        # img = img.resize((350, 350))
        img = img.convert('RGB')
        img = np.array(img)
        fig,ax= plt.subplots(1)
        ax.imshow(img)
        boxes, labels, probs = predictor.predict(img, 500 , .99)
        np_boxes = boxes.cpu().detach().numpy()
        #save  to dict that can  be writen as json the boxess and probs
        if len(np_boxes)==0:
            output_dict[image][0]=[]
        else:
            bb_height = np_boxes[:, 3] - np_boxes[:, 1]
            bb_width = np_boxes[:, 2] - np_boxes[:, 0]
            np_boxes = augmented_cut(np_boxes, bb_width,bb_width,leeway)
            count= 0
            for box in np_boxes:

                #crop  the image
                crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                #resize to 256,256
                crop_img = cv2.resize(crop_img, (256, 256))
                # crop_img = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
                # plt.imshow(crop_img)
                # plt.show()
                #resize the image
                #convert the image to a tesnor of shape  (1,256,256,3)
                tf_img= tf.keras.preprocessing.image.img_to_array(crop_img)
                tf_img= tf_img.reshape((1,256,256,3))
                #predict the image
                yaw_pred, pitch_pred, roll_pred = model(tf_img,training=False)
                yaw_pred,pitch_pred,roll_pred=tf.math.softmax(yaw_pred),tf.math.softmax(pitch_pred),tf.math.softmax(roll_pred)
                output_dict[image][count] = {

                    'box': box.tolist()}
                yaw_pred=tf.reduce_sum(tf.multiply(yaw_pred,idx_tensor))*3-99
                pitch_pred=tf.reduce_sum(tf.multiply(pitch_pred,idx_tensor))*3-99
                roll_pred=tf.reduce_sum(tf.multiply(roll_pred,idx_tensor))*3-99
                output_dict[image][count]['label']=[yaw_pred.numpy(),pitch_pred.numpy(),roll_pred.numpy()]
                count= count+1
    except:
        pass
#%%
import pickle
with open('output_dict.pkl', 'wb') as f:
    pickle.dump(output_dict, f)
