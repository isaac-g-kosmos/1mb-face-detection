import numpy as np
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.config.fd_config import define_img_size
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
define_img_size(320)
leeway = .2

# model_path = "models/pretrained/version-RFB-640.pth"
#
# net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
# net.load(model_path)                              weq
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)

# model_path = "models/pretrained/version-slim-320.pth"
model_path = "/home/ubuntu/1mb-face-detection/models/pretrained/version-RFB-320.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)

net.load(model_path)
# %%
initial_path="/home/ubuntu/spoof-detection/linux_datasets/with_bb/CELEB_train_1.csv"
df = pd.read_csv(initial_path)
# df['path'] = df['path'].apply(lambda x: x[1:])
data_list = df['path'].tolist()
del df
# %%
dictionary = {}

for x in data_list:
    boxes_pic = []
    path = x

    img = Image.open(path)
    width = img.width
    height = img.height
    img = img.convert('RGB')
    img = np.array(img)

    boxes, labels, probs = predictor.predict(img, 1000 / 2, .9)
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    # plt.show()
    np_boxes = boxes.cpu().detach().numpy()
    for box in np_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
        box1 = [box[0], box[1], box[2], box[3]]
        boxes_pic.append(box1)
        # ax.add_patch(rect)
    # plt.show()
    # break
    dictionary[x] = {"boxes":boxes_pic,"width":width,"height":height}
# %%
number_list = []
for x in dictionary.keys():
    number_list.append(len(dictionary[x]))
print(max(number_list))
print(min(number_list))
print(np.mean(number_list))

# make a df with images with only one pic their path box and label
df1 = pd.DataFrame(columns=['path', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2',"width","height"])
for x in dictionary.keys():
    if len(dictionary[x]["boxes"]) == 1:
        df1 = pd.concat([df1, pd.DataFrame({
            'path': [x],
            'bb_x1': [box[0]],
            'bb_y1': [box[1]],
            'bb_x2': [box[2]],
            'bb_y2': [box[3]],
            'width': [dictionary[x]['width']],
            'height': [dictionary[x]['height']]
        })], ignore_index=True)
df1.to_csv(f"/home/ubuntu/1mb-face-detection/spoof_utils/{os.path.basename(initial_path)}",index=False)
# %%
