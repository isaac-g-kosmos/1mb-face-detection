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
model = tf.keras.models.load_model(
    r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\video_inference_data\movnet_sparkling-moon-39.h5")
# %%
img_path = r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images'

images= os.listdir(img_path)
#%%
output_dict= {}
#%%
for image in images:
    output_dict[image] = {}
    img=Image.open(os.path.join(img_path,image))
    #down  sample to 500X500
    # img = img.resize((350, 350))
    img = img.convert('RGB')
    img = np.array(img)
    fig,ax= plt.subplots(1)
    ax.imshow(img)
    boxes, labels, probs = predictor.predict(img, 500 , .8)
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
            landmarks, glasses, lighting, face, pose, hat= model.predict(tf_img)
            #save the image
            output_dict[image][count]= {'labels':[landmarks[0].tolist(), glasses[0].tolist(), lighting[0], face[0], pose[0], hat[0]],
                                 'box':box.tolist()}

            count= count+1
#%%
import pickle
with open('output_dict.pkl', 'wb') as f:
    pickle.dump(output_dict, f)
