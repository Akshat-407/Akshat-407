import cv2
import numpy as np
import pandas as pd
import torch
from src.data_loader.basedataloader import BaseDataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_loader.mammo_transforms import TrainTransformSupCon, TwoCropTransform
import matplotlib.pyplot as plt

class MammoSupervisedTest(BaseDataloader):
    def __init__(self, csv_file_valid,input_size,input_size_two, batch_size,method, sampler, num_workers, shuffle):
        super().__init__()
        self.csv_file_valid = csv_file_valid
        self.input_size = input_size
        self.input_size_two = input_size_two
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.method = method
        
        
        ##Valid File
        self.val_read_file = pd.read_csv(self.csv_file_valid)
        #abnormality_val = self.val_read_file['abnormality type']
        pathology_val= self.val_read_file['subtype']
        #combined_val = abnormality_val.astype(str)   + '_' + pathology_val.astype(str)
        self.val_img_path = self.val_read_file['CC_PNG'].tolist() + self.val_read_file['MLO_PNG'].tolist()
        self.abnormality_val = self.val_read_file['abnormality'].tolist() + self.val_read_file['abnormality'].tolist()
        self.valid_labels = pathology_val.tolist()+pathology_val.tolist()

        self.eval_imgs = []
        self.labels_valid = []
        self.files = []
        self.abn =[]
        for indexs, eval_f in enumerate(self.val_img_path):
            with open(eval_f, 'rb') as f:
                label_temp_val = self.class_name_to_labels(self.valid_labels[indexs])
                self.labels_valid.append(label_temp_val)
                self.eval_imgs.append(f.read())
                self.files.append(eval_f)
                self.abn.append(self.abnormality_val[indexs])
        self.setup_datset()

    def class_name_to_labels(self, class_name):

        if class_name == 'Luminal A' or class_name == 'Luminal B':
            labels = 0.0
        elif class_name == 'HER2-enriched' or class_name == 'triple negative':
            labels = 1.0
        return labels

    def setup_datset(self):
        

        valid_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((self.input_size, self.input_size_two)), transforms.ToTensor()])
        
        
        
        self.dataloader_eval = SingleImgDataset(self.eval_imgs, self.labels_valid, self.files,self.abn,  transforms=valid_transform)

    def test_dataloader(self):
        return DataLoader(self.dataloader_eval, batch_size=self.batch_size, sampler = None, num_workers=self.num_workers, shuffle=self.shuffle)


class SingleImgDataset:
    def __init__(self, image_list, label_list,file_list,abn_list, transforms):
        self.image_list = image_list
        self.label_list = label_list
        self.file_list = file_list
        self.transforms = transforms
        self.abn_list = abn_list
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img = cv2.imdecode(np.frombuffer(self.image_list[i], dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        img = img * 1.0 / img.max()
        
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)

        if self.transforms:
            img = self.transforms(img)    
        label = self.label_list[i]
        file = self.file_list[i]
        abn = self.abn_list[i]
        return img, label, file, abn
        
        


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]