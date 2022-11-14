import pandas as pd
import numpy as np
import pydicom
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import cv2
import glob


import cv2

import numpy as np

import albumentations as A
import pandas as pd
import torchvision
import torch
import torch
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from src.data_loader.base_dataset import BaseDataset
from src.data_loader.basedataloader import BaseDataloader
from torch.utils.data import DataLoader
from torchvision import transforms


class Mammo(BaseDataloader):
    def __init__(self,batch_size,input_size,input_size_two):
        super().__init__()
        #self.mammography = pd.read_csv(excel_file, sep=';')


        #self.train_folder = excel_file + "/train"  # train.csv
        #self.eval_folder = excel_file + "/eval"  # valid,csv
        #self.train_read_file = pd.read_csv(r'C:\Users\aksha\PycharmProjects\Thesis\CBIS_Analysis\gain_mammo_new-main\gain_mammo_new-main\Labels.csv', skipinitialspace = True, delimiter = ";")
        #self.val_read_file = pd.read_csv(r'C:\Users\aksha\PycharmProjects\Thesis\CBIS_Analysis\gain_mammo_new-main\gain_mammo_new-main\Labels.csv',  skipinitialspace = True, delimiter = ";")
        self.train_read_file = pd.read_csv(r'/cluster/shared_dataset/CBIS-DDSM/Data/Labels/calcification_labels/train_CV1.csv',skipinitialspace=True, delimiter=";")
        self.val_read_file = pd.read_csv(r'/cluster/shared_dataset/CBIS-DDSM/Data/Labels/calcification_labels/valid_CV1.csv',skipinitialspace=True, delimiter=";")
        self.train_img_path = self.train_read_file['image file path'].tolist()
        self.train_labels = self.train_read_file['pathology'].tolist()

        self.val_img_path = self.val_read_file['image file path'].tolist()
        self.valid_labels = self.val_read_file['pathology'].tolist()
        # image_file=pandas.r

        self.batch_size = batch_size
        self.input_size= input_size
        self.input_size_two=input_size_two
        #self.patch_size = patch_size

        self.train_imgs = []
        self.labels_train=[]
        count=0
        for index, train_f in enumerate(self.train_img_path):



            with open(train_f, 'rb') as f:

                #print(self.train_labels[count])
                label_temp=self.class_name_to_labels(self.train_labels[index])
                #print(label_temp)
                #print(index)
                self.labels_train.append(label_temp)
                self.train_imgs.append(f.read())


        self.eval_imgs = []
        self.labels_valid = []
        count = 0
        for indexs, eval_f in enumerate(self.val_img_path):

            count += 1
            with open(eval_f, 'rb') as f:
                label_temp_val=self.class_name_to_labels(self.valid_labels[indexs])
                self.labels_valid.append(label_temp_val)
                self.eval_imgs.append(f.read())

        self.setup_datset()


    def class_name_to_labels(self,class_name):
        if class_name == 'MALIGNANT':
            labels = 0.0
        elif class_name == 'BENIGN':
            labels = 1.0
        elif class_name == 'BENIGN_WITHOUT_CALLBACK':
            labels = 2.0

        return labels


    def setup_datset(self):

        #train_transforms = [A.Resize(32,32),A.GaussNoise(), A.GaussianBlur(),, A.Equalize(p=0.4), A.HorizontalFlip(p=0.5), A.VerticalFlip(), A.CenterCrop(1024,512)
        #                   A.RandomBrightnessContrast(), A.HorizontalFlip(), A.VerticalFlip(),A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        #train_transforms=[A.Resize(self.input_size,self.input_size_two),  A.HorizontalFlip(), A.VerticalFlip(),#A.Equalize(p=0.5),
                          #A.CoarseDropout (max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
                          #A.Rotate (limit=180)]
        #train_transforms = []

        #train_transform = A.Compose(train_transforms)
        train_transform= transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size_two)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomEqualize(p=0.4),
                #transforms.RandomApply(torch.nn.ModuleList([
                #    transforms.RandomRotation(20), # look..
                #]), p=0.5), ## can experiment# remove
                transforms.RandomAffine(degrees=(-25,25),scale=(0.8,1.2), shear=0.12), # or use shear=0.12
                transforms.ToTensor(),

                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])

        valid_transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((self.input_size,self.input_size_two)),transforms.ToTensor()])



        self.dataloader_train = SingleImgDataset(self.train_imgs,self.labels_train,transforms=train_transform)
        self.dataloader_eval = SingleImgDataset(self.eval_imgs,self.labels_valid,transforms= valid_transform)

    def train_dataloader(self):
        return DataLoader(self.dataloader_train, batch_size=self.batch_size,num_workers=8, shuffle= True)

    def val_dataloader(self):
        return DataLoader(self.dataloader_eval, batch_size=self.batch_size,num_workers=8, shuffle= True)




class SingleImgDataset:
    def __init__(self,image_list,label_list,transforms):
        self.image_list= image_list
        self.label_list= label_list
        self.transforms= transforms
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self,i):
        img = cv2.imdecode(np.frombuffer(self.image_list[i], dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]

        #print(img.dtype)
        #print(img.shape)
        #img= torchvision.transforms.Normalize(img)
        img = img * 1.0 / img.max()
        img=torch.from_numpy(img.astype(np.float32))
        #img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img=img.permute(2,0,1)
        #print(img.shape)

        if self.transforms:
            img = self.transforms(img)
            #    img = self.transforms(image=img) # for albumentation
        #print(img)

        #print(img.dtype)
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()

        label = self.label_list[i]

        return img,label #torch.from_numpy(img.copy()), label     
    


'''
import matplotlib.pyplot as plt
data=Mammo()
train=data.train_dataloader()
for images, labels in train:
    #img = images.cpu().detach().numpy()
    plt.imshow(list(images.values())[0][0])
    plt.show()
    print(labels)
    
'''




class MammographyCBISDDSM(Dataset):

    def __init__(self, excel_file, category,image_height, image_width, transform=None):
        """
        Args:
            excel_file (string): Path to the excel file with annotations.
            category (string) : 'Classification' for Benign and Malignant. 'Subtypes' for Subtype Classification
            transform (callable, optional): Optional transform to be applied
        """
        self.mammography = pd.read_csv(excel_file,sep = ';')
        self.category = category
        self.transform = transform
        self.image_height=image_height
        self.image_width=image_width

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, idx):
        if self.category == 'Classification':
            class_name = self.mammography.iloc[idx, 9]
            if class_name == 'MALIGNANT':
                labels = 0.0
            elif class_name == 'BENIGN':
                labels = 1.0
            elif class_name == 'BENIGN_WITHOUT_CALLBACK':
                labels = 2.0

        return labels


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = self.mammography.iloc[idx, 11]
        mask_folder= self.mammography.iloc[idx, 12]
        file_name= os.path.basename(img_folder)

        #print(img_folder)
        #print(os.path.exists(img_folder))
        #image_array = pydicom.dcmread(img_folder)
        #for img in glob.glob(img_folder):# TO DO: Using CV2
        image_array=cv2.imread(img_folder)
        image_array= cv2.resize(image_array, (self.image_height,self.image_width))
        mask_array = cv2.imread(mask_folder)
        mask_array=cv2.resize(mask_array, (self.image_height,self.image_width))
        #image_array = image_array.pixel_array
        image_array = image_array * 1.0 / image_array.max()
        #image_array = image_array * (1.0 /255.0)
        #plt.imshow(image_array)
        #print(image_array.shape)
        #plt.show()
        image_array = torch.from_numpy(image_array.astype(np.float32))
        mask_array = torch.from_numpy(mask_array)
        #image_array = image_array.repeat(3, 1, 1)
        #print(image_array.shape)

        if self.transform:
            # original_tensor_dim = [256, 200, 4]
            image_array = image_array.permute(2,0,1)
            mask_array = mask_array.permute(2, 0, 1)
            #print(image_array.shape)
            # changed_tensor_dim = [4, 256, 200]
            image_array,mask_array = self.transform(image_array,mask_array)
            #print(image_array.shape)
            #print('')

        
        labels = self.class_name_to_labels(idx)
        labels = torch.from_numpy(np.array(labels))
        return image_array, labels, mask_array,file_name


class MammographyCBISDDSMGrad(Dataset):

    def __init__(self, excel_file, category, transform=None):
        """
        Args:
            excel_file (string): Path to the excel file with annotations.
            category (string) : 'Classification' for Benign and Malignant. 'Subtypes' for Subtype Classification
            transform (callable, optional): Optional transform to be applied
        """
        self.mammography = pd.read_csv(excel_file, sep=';')
        self.category = category
        self.transform = transform

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, idx):
        if self.category == 'Classification':
            class_name = self.mammography.iloc[idx, 9]
            if class_name == 'MALIGNANT':
                labels = 0.0
            elif class_name == 'BENIGN':
                labels = 1.0
            elif class_name == 'BENIGN_WITHOUT_CALLBACK':
                labels = 2.0

        return labels

    def image_cropper(self, original_image_array, seg_array): # Black pixels remover
        column_indices = []
        for i in range(0, original_image_array.shape[1]):
            pixel_columns = original_image_array[:, i]
            pixel_columns = np.array(pixel_columns)
            if np.count_nonzero(pixel_columns) == 0:
                column_indices.append(i)

        row_indices = []
        for j in range(0, original_image_array.shape[0]):
            pixel_rows = np.array(original_image_array[j, :])
            if np.count_nonzero(pixel_rows) == 0:
                row_indices.append(j)

        edited_image = np.delete(original_image_array, column_indices, axis=1)
        final_edited_image = np.delete(edited_image, row_indices, axis=0)
        seg_edited_image = np.delete(seg_array, column_indices, axis=1)
        seg_final_edited_image = np.delete(seg_edited_image, row_indices, axis=0)


        return seg_final_edited_image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = self.mammography.iloc[idx, 11]
        
        dcm_folder = self.mammography.iloc[idx, 14]
        seg_mask = self.mammography.iloc[idx, 13]

        dcm_array = pydicom.dcmread(dcm_folder)
        dcm_array = dcm_array.pixel_array

        seg_array = pydicom.dcmread(seg_mask)
        seg_array = seg_array.pixel_array
         
        seg_array = self.image_cropper(dcm_array, seg_array)

        # print(img_folder)
        # print(os.path.exists(img_folder))
        # image_array = pydicom.dcmread(img_folder)
        # for img in glob.glob(img_folder):# TO DO: Using CV2
        image_array = cv2.imread(img_folder)
        # image_array = image_array.pixel_array
        image_array = image_array * 1.0 / image_array.max()
        # image_array = image_array * (1.0 /255.0)
        # plt.imshow(image_array)
        # print(image_array.shape)
        # plt.show()
        image_array = torch.from_numpy(image_array.astype(np.float32))
        seg_array = torch.from_numpy(seg_array.astype(np.float32))
        #image_array = image_array.repeat(3, 1, 1)
        #seg_array = seg_array.repeat(3, 1, 1)
        # print(image_array.shape)
        #seg_array = cv2.resize(seg_array,(1024,2048))
        if self.transform:
            # original_tensor_dim = [256, 200, 4]
            image_array = image_array.permute(2, 0, 1)
            # print(image_array.shape)
            # changed_tensor_dim = [4, 256, 200]
            image_array = self.transform(image_array)
            seg_array = self.transform(seg_array)            
            # print(image_array.shape)
            # print('')

        labels = self.class_name_to_labels(idx)
        labels = torch.from_numpy(np.array(labels))
        return image_array, labels, seg_array