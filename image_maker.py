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
define_img_size(320)

model_path = "models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)
leeway = 0.2

model = tf.keras.models.load_model(
    r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\video_inference_data\movnet_sparkling-moon-39.h5")
# %%
targe_path = r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images'


vid_path = r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\video_inference_data\Interview with a Postdoc, Junior Python Developer.mp4'

vidcap = cv2.VideoCapture(vid_path)
success, image = vidcap.read()
count = 0
while success:
    #convert image to rbg
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #save the  image
    cv2.imwrite(os.path.join(targe_path, f"{count}.jpg"), image)
    count += 1
    print(count)
    success, image = vidcap.read()

