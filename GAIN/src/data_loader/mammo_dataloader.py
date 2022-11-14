import cv2
import numpy as np
import pandas as pd
import torch
from src.data_loader.basedataloader import BaseDataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_loader.mammo_transforms import TrainTransformSupCon, TwoCropTransform
import matplotlib.pyplot as plt

class MammoSupervised(BaseDataloader):
    def __init__(self, csv_file_train, csv_file_valid,input_size,input_size_two, batch_size,method, sampler, num_workers, shuffle):
        super().__init__()

        self.csv_file_train = csv_file_train
        self.csv_file_valid = csv_file_valid
        self.input_size = input_size
        self.input_size_two = input_size_two
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.method = method
        
        
        ##Train File
        self.train_read_file = pd.read_csv(self.csv_file_train)
        self.train_img_path = self.train_read_file['filename'].tolist()
        pathology = self.train_read_file['Pathology Classification/ Follow up']
        self.train_labels = pathology.tolist()
        self.train_seg_mask = self.train_read_file['Segmentation_Mask'].tolist()
        
        ##Valid File
        self.val_read_file = pd.read_csv(self.csv_file_valid)
        self.val_img_path = self.val_read_file['filename'].tolist()
        pathology_val= self.val_read_file['Pathology Classification/ Follow up']
        self.valid_labels = pathology_val.tolist()
        self.valid_seg_mask = self.val_read_file['Segmentation_Mask'].tolist()

        self.train_imgs = []
        self.labels_train = []
        self.train_segs=[]
        for index, train_f in enumerate(self.train_img_path):
            with open(train_f, 'rb') as f:
                label_temp = self.class_name_to_labels(self.train_labels[index])
                self.labels_train.append(label_temp)
                self.train_imgs.append(f.read())
                with open(self.train_seg_mask[index], 'rb') as fs:
                    self.train_segs.append(fs.read())

        self.eval_imgs = []
        self.labels_valid = []
        self.valid_segs=[]
        for indexs, eval_f in enumerate(self.val_img_path):
            with open(eval_f, 'rb') as f:
                label_temp_val = self.class_name_to_labels(self.valid_labels[indexs])
                self.labels_valid.append(label_temp_val)
                self.eval_imgs.append(f.read())
                with open(self.valid_seg_mask[indexs], 'rb') as fss:
                    self.valid_segs.append(fss.read())
        self.setup_dataset()

    def get_path_names(self, pathname,listname):
        filenames = []
        for i in listname:
            filename = pathname + listname + '.jpg'
            filenames.append(filename)
        return filenames
        
        
    def class_name_to_labels(self, class_name):

        if class_name == 'Normal':
            labels = 0.0
        elif class_name == 'Benign':
            labels = 1.0
        elif class_name == 'Malignant':
            labels = 2.0
        return labels

    def setup_dataset(self):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((self.input_size, self.input_size_two), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomEqualize(p=0.2),
            #transforms.RandomAffine(degrees=(-25, 25), scale=(0.8, 1.2), shear=0.12),  # or use shear=0.12
            transforms.ToTensor(),
            #transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 2.3), value=0, inplace=False)
        ])

        valid_transform = transforms.Compose(
            [transforms.ToPILImage(),transforms.ToTensor()])
        if self.method == 'SupCon':
            self.dataloader_train = SingleImgDataset(self.train_imgs, self.labels_train, self.train_segs, transforms=TwoCropTransform(train_transform))
        else:
            self.dataloader_train = SingleImgDataset(self.train_imgs, self.labels_train, self.train_segs,  transforms=train_transform)
        self.dataloader_eval = SingleImgDataset(self.eval_imgs, self.labels_valid, self.valid_segs, transforms=train_transform)

    def train_dataloader(self):
        return DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler = self.sampler, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataloader_eval, batch_size=self.batch_size, sampler = None, num_workers=self.num_workers, shuffle=False)


class SingleImgDataset:
    def __init__(self, image_list, label_list,seg_list, transforms):
        self.image_list = image_list
        self.label_list = label_list
        self.seg_list = seg_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img = cv2.imdecode(np.frombuffer(self.image_list[i], dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        #img = img * 1.0 / img.max()
        img = cv2.resize(img, (224,224))
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)
        seg = cv2.imdecode(np.frombuffer(self.seg_list[i], dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        seg = cv2.resize(seg, (224,224))#[None, ...]
        seg = torch.from_numpy(seg.astype(np.float32))
        seg = seg.permute(2, 0, 1)
        if self.transforms:
            img = self.transforms(img)
            seg = self.transforms(seg) 
        label = self.label_list[i]

        return img, label, seg


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]