import numpy as np
import pandas as pd
# from sklearn.metrics import classification_report
from PIL import Image
import tensorflow as tf
import wandb
# from tensorflow_addons.metrics import MultiLabelConfusionMatrix
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def augmented_cut(BB, width, height, leeway=0):
    BB[:, 0] = BB[:, 0] - leeway * width / 2
    BB[:, 1] = BB[:, 1] - leeway * height / 2
    BB[:, 2] = BB[:, 2] + leeway * width
    BB[:, 3] = BB[:, 3] + leeway * height

    return BB


