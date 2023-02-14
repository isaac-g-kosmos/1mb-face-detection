import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

from vision.ssd.config.fd_config import define_img_size

import torch
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(320)

# model_path = "models/pretrained/version-RFB-640.pth"
#
# net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
# net.load(model_path)
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)

# model_path = "models/pretrained/version-slim-320.pth"
model_path = "models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)

net.load(model_path)
# %%
model = tf.keras.models.load_model(
    r'C:\Users\isaac\PycharmProjects\tensorflow_filter\models\movnet_stellar-dragon-11.h5')
# %%
stock_images = os.listdir(r'C:\Users\isaac\PycharmProjects\sat-robot\stock_images')
stock_images = [os.path.join(r'C:\Users\isaac\PycharmProjects\sat-robot\stock_images', x) for x in stock_images]
# %%
random_image = random.choice(stock_images)
# img = Image.open(random_image)
# img = img.convert('RGB')
# img = np.array(img)
# boxes, labels, probs = predictor.predict(img, 1000 / 2, .95)
# fig, ax = plt.subplots(1)
# ax.imshow(img)
# np_boxes = boxes.cpu().detach().numpy()
# for box in np_boxes:
#     rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
#     ax.add_patch(rect)
# plt.show()
# plt.imshow(slice)
# plt.show()
# # %%
# slice = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
# # convert slice to tensor
# slice = np.expand_dims(slice, axis=0)
# # slice=np.expand_dims(slice,axis=3)
# slice = slice / 255
# slice = slice.astype(np.float32)
# # %%
# # resize to 256x256
# slice = tf.image.resize(slice, [256, 256])


class InferenceWrapper:
    def __init__(self, model_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        define_img_size(320)
        face_model_path = "models/pretrained/version-RFB-320.pth"
        net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)
        net.load(face_model_path)
        self.predictor = predictor
        model = tf.keras.models.load_model(model_path)
        self.model = model

    def pre_procees(self, img, box):
        img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        img=np.expand_dims(img, axis=0)
        # img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32)
        img = img / 255.
        img = tf.image.resize(img, [256, 256])

        return img

    def predict(self, img):
        boxes, labels, probs = self.predictor.predict(img, 1000 / 2, .95)
        np_boxes = boxes.cpu().detach().numpy()
        for box in np_boxes:
            img = self.pre_procees(img, box)
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict(img)
            print(pred)
            return pred

    def model_predict(self, img):
        landmarks, glasses, lighting, face, pose, hat = self.model(img)
        # dictionary_output = {'landmarks': landmarks,
        #                      'glasses': glasses,
        #                      'lighting': lighting,
        #                      'face': face,
        #                      'pose': pose,
        #                      'hat': hat}

        return landmarks, glasses, lighting, face, pose, hat

    # def pose_inference_img(self, img,pose):

    def load_and_preprocess_image(path):
        image_string = tf.io.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string,
                                             channels=3)  # Channels needed because some test images are b/w
        # print(image_decoded.shape)
        image_resized = tf.image.resize(image_decoded, [720, 1280])
        return tf.cast(image_resized, tf.float32)

    def frame_inference(self, img):

        img=Image.open(img)
        img = img.convert('RGB')
        img=np.array(img)
        boxes, labels, probs = self.predictor.predict(img, 1000 / 2, .95)
        np_boxes = boxes.cpu().detach().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for box in np_boxes:
            img_slice = self.pre_procees(img, box)
            inference_img=np.array(img_slice[0])
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(inference_img)
            fig1.show()
            # img_slice = np.expand_dims(img_slice, axis=0)
            landmarks, _, _, _, pose, _ = self.model_predict(img_slice)
            #plot landmarks



            if pose[0][0] < 0.5:
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='green')
                anotation = plt.annotate('Good', (box[0], box[1]), color='red')
            else:
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
                anotation = plt.annotate('Bad', (box[0], box[1]), color='red')

            ax.add_patch(rect)
        fig.show()
        return ax,fig
Wrapper= InferenceWrapper(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\models\movnet_stellar-dragon-11.h5')
# random_image = random.choice(stock_images)
# img = Image.open('PattyHearstmug.jpg')
# img = img.convert('RGB')
# img = np.array(img)
ax,fig=Wrapper.frame_inference('PattyHearstmug.jpg')