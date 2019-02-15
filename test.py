import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from utils import TrainTransform, ValTransform
from data import MyDataset
from data.config import mydataset as cfg
from models.model_builder import model_builder
import pandas as pd
from data import *
from tqdm import tqdm
import numpy as np
import time
import os 
import sys
import json
from utils import transforms


def arg_parse():

    parser = argparse.ArgumentParser(
        description='Mydataset classification')
    parser.add_argument('-v', '--version', default='resnet50',
                        help='')
    parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
    parser.add_argument('--weights', '--weights',default="./weights/resnet50_best_model.pth", help='weights path')
    args = parser.parse_args()

    return args

def main():
    args = arg_parse()
    model_name = args.version
    weights = args.weights
    num_classes = cfg["num_classes"]
    img_hw = cfg["img_hw"]
    bgr_means = cfg["bgr_means"]
    Testroot=cfg['Testroot']
    net = model_builder(model_name, pretrained=False, num_classes=num_classes)
    net.cuda()
    state_dict = torch.load(weights)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    test_transform = ValTransform(img_hw, bgr_means, padding=False)
    # test_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384, 384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])
    test_dataset = MyWhaleDataset(Testroot, 'test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=
                                              args.num_workers)
    test_net(test_loader, net)

def test_net(test_loader, net):
    train_df = pd.read_csv("/home/siting/files/DingGuo/whaledataset/train.csv")
    print(train_df.head())
    y, lab_encoder = prepare_labels(train_df['Id'])

    sub = pd.read_csv('/home/siting/files/DingGuo/whaledataset/sample_submission.csv')
    net.eval()
    for (data, target, name) in tqdm(test_loader):
        data = data.cuda()
        output = net(data)
        output = output.cpu().detach().numpy()
        for i, (e, n) in enumerate(list(zip(output, name))):
            sub.loc[sub['Image'] == n, 'Id'] = ' '.join(lab_encoder.inverse_transform(e.argsort()[-5:][::-1]))

    sub.to_csv('submission.csv', index=False)
    print("done! The submission.csv is written.")

if __name__ == '__main__':
    main()
