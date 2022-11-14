import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import re
import matplotlib.pyplot as plt
import glob
import os
import random
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class MammographyDataset(Dataset):

    def __init__(self, file, data_folder, keyword, image_height, image_width, img_col, seg_col, lab_col,task, transform=None):
        """
        Args:
            file (string): Path to the file with annotations.
            transform (callable, optional): Optional transform to be applied
        """
        self.mammography = pd.read_csv(file)
        self.data_folder = data_folder
        self.keyword = keyword
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.img_col = img_col
        self.seg_col = seg_col
        self.lab_col = lab_col
        self.task = task

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, class_name):

        if class_name == 'Normal':
            labels = 0.0
        elif class_name == 'Benign':
            labels = 1.0
        elif class_name == 'Malignant':
            labels = 2.0
        return labels

    def class_name_to_ohe(self, class_name):

        if class_name == 'Normal':
            labels = [1.,0.,0.]
        elif class_name == 'Benign':
            labels = [0.,1., 0.]
        elif class_name == 'Malignant':
            labels = [0.,0.,1.]
        return labels

    def image_load(self, idx, column):
        img_name = self.mammography.iloc[idx, column]
        part_after = img_name.split("\\images\\")[-1]
        img_name = os.path.join(self.data_folder, part_after)
        image = cv2.imread(img_name)
        image = cv2.resize(image, (self.image_height, self.image_width))
        image = torch.from_numpy(image)#.astype(np.float32))
        image = image.permute(2, 0, 1)
        return image, part_after

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, part_after = self.image_load(idx, self.img_col)
        mask,_ = self.image_load(idx, self.seg_col)
        labels = self.mammography.iloc[idx, self.lab_col]

        if self.transform and self.task in ['gains','segmentation','pix2pix']:
            image, mask = self.transform(image, mask)
        else:
            image = self.transform(image)

        if self.task == 'gain':
            labels = self.class_name_to_ohe(labels)
        else:
            labels = self.class_name_to_labels(labels)
        labels = torch.from_numpy(np.array(labels))
        return image, labels, mask, part_after




class CycleGANDataset(Dataset):
    def __init__(self,file_A ,file_B, transform=None, unaligned=False,column = 3, mode="train"):

        self.transform = transform
        self.unaligned = unaligned
        self.column = column
        self.files_A = pd.read_csv(file_A)
        self.files_B = pd.read_csv(file_B)


    def __getitem__(self, index):

        image_A = Image.open(self.files_A.iloc[index%len(self.files_A), self.column])#.convert('L')
        image_A = image_A.resize((224,224))
        image_A =np.array(image_A)
        image_A = (image_A - image_A.min()) / (image_A.max() - image_A.min())
        item_A = self.transform(image_A)
        if self.unaligned:
            image_B = Image.open(self.files_B.iloc[random.randint(0, len(self.files_B) - 1), self.column])#.convert('L')
        else:
            image_B = Image.open(self.files_B.iloc[index%len(self.files_A), self.column])#.convert('L')
        image_B = image_B.resize((224, 224))
        image_B = np.array(image_B)
        image_B = (image_B - image_B.min()) / (image_B.max() - image_B.min())
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))