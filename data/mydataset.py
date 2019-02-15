# write by yqyao
# 
import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from data.config import mydataset as cfg
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MyDataset(data.Dataset):

    def __init__(self, root, phase, tarnsform=None, target_transform=None,
                 dataset_name='MyDataset'):
        self.root = root
        self.phase = phase
        self.tarnsform = tarnsform
        self.target_transform = target_transform
        self.name = dataset_name
        self.images_targets = list()
        data_list = os.path.join(root, self.phase + '.txt')
        with open(data_list ,"r") as f:
            for line in f.readlines():
                line = line.strip().split()
                if self.phase == 'test':
                    self.images_targets.append((line[0], 0))
                else:    
                    self.images_targets.append((line[0], int(line[1])))

    def __getitem__(self, index):
        img_id = self.images_targets[index]
        target = img_id[1]
        path = img_id[0]
        # img = Image.open(path).convert('RGB')
        img = cv2.imread(path)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.tarnsform is not None:
            img = self.tarnsform(img)
        return img, target

    def __len__(self):
        return len(self.images_targets)


class MyWhaleDataset(data.Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform=None, y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train' or self.datatype=='val':
            self.df = df.values
            self.image_files_list = [s for s in os.listdir(os.path.join(datafolder,'train'))]
        else:
            self.image_files_list = [s for s in os.listdir(os.path.join(datafolder, datatype))]
        self.transform = transform

    def __len__(self):
        if self.datatype == 'train' or self.datatype =='val':
            return len(self.df)
        else:
            return len(self.image_files_list)

    def __getitem__(self, idx):
        if self.datatype == 'train' or self.datatype =='val':
            img_name = os.path.join(self.datafolder, 'train', self.df[idx][0])
            label = self.y[idx]

        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.datatype, self.image_files_list[idx])
            label = np.zeros((cfg['num_classes'],))

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)
        if self.datatype == 'train'or self.datatype =='val':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]

def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder
