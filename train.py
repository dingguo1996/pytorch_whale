import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
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
from data import *
from data.config import mydataset as cfg
from models.model_builder import model_builder
import numpy as np
from tensorboardX import SummaryWriter
import time
import os
import re
import sys
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from utils import AverageMeter
from utils import transforms
# import setproctitle
# setproctitle.setproctitle("python")

def arg_parse():

    parser = argparse.ArgumentParser(
        description='Mydataset classification')
    parser.add_argument('-v', '--version', default='resnet50',
                        help='')
    parser.add_argument('-b', '--batch_size', default=64,
                        type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4,
                        type=int, help='Number of workers used in dataloading')
    parser.add_argument('--iscuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate',
                        default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--ngpu', default=2, type=int, help='gpus')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument(
        '--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0,
                        type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1,
                        type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')
    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size, iteration, lr):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch < 1:
        iteration += iteration * epoch
        lr = 1e-6 + (lr - 1e-6) * iteration / (epoch_size * 1)
    else:
        div = 0
        if epoch >= step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch >= step_epoch[idx] and epoch < step_epoch[idx+1]:
                    div = idx 
                    break
        lr = lr * (gamma ** div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, lr, writer):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    pbar=tqdm(train_loader)
    iteration=0
    # for iteration, (img, target) in enumerate(train_loader):
    for img, target in pbar:
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma, epoch_size, iteration, lr)
        img = img.cuda()
        target = target.float().cuda()
        t0 = time.time()
        out = net(img)
        t1 = time.time()
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if iteration % 10 == 0:
            print("Epoch: {} | iter {}/{} loss: {} Time: {} lr: {}".format(str(epoch), str(iteration), str(epoch_size), str(round(loss.item(), 5)), str(round(t2 - t0, 5)), str(lr)))
            writer.add_scalar('loss', loss, epoch*epoch_size+iteration)
        pbar.set_description(desc='train')
        iteration+=1

def save_checkpoint(net, epoch, is_best, save_folder, model_name):
    file_name = os.path.join(save_folder, model_name + "_epoch_{}".format(str(epoch))+ '.pth')
    torch.save(net.state_dict(), file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_folder, model_name+"_best_model.pth"))

def resume_checkpoint(net, save_folder, start_epoch):
    files = os.listdir(save_folder)
    for f in files:
        iter_string = re.findall(r'\S+epoch_(\d+)\.pth', f)
        if len(iter_string) > 0:
            checkpoint_iter = int(iter_string[0])
            if checkpoint_iter > start_epoch:
                # Start one iteration immediately after the checkpoint iter
                start_epoch = checkpoint_iter + 1
                resume_weights_file = f
    if start_epoch > 0:
        # Override the initialization weights with the found checkpoint
        weights_file = os.path.join(save_folder, resume_weights_file)
        print('========> Resuming from checkpoint {} at start iter {}'.
              format(weights_file, start_epoch))

        state_dict = torch.load(weights_file)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    return net,start_epoch


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    target_id=torch.argmax(target,dim=1)
    pred=torch.argmax(output,dim=1)
    res=0
    for i in range(batch_size):
        if target_id[i]==pred[i]:
            res+=1

    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    #
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #     res_app = correct_k.mul_(100.0 / batch_size)
    #     res.append(res_app.cpu().numpy())
    return res

def eval_net(val_loader, net, criterion):
    prec_sum = 0
    pbar=tqdm(val_loader)
    pbar.set_description(desc='validation')
    for img, target in pbar:
        with torch.no_grad():
            target = target.cuda()
            img = img.cuda()
            img_var = torch.autograd.Variable(img)
            target_var = torch.autograd.Variable(target)
            output = net(img_var)
            loss = criterion(output.double(), target_var.double())
            prec = accuracy(output.data, target)
            prec_sum += prec
    return prec_sum, loss.item()

def last_weight_test_net(test_loader, net, lab_encoder):
    sub = pd.read_csv('/home/siting/files/DingGuo/whaledataset/sample_submission.csv')
    pbar=tqdm(test_loader)
    pbar.set_description(desc='test')
    net.eval()
    for (data, target, name) in pbar:
        data = data.cuda()
        output = net(data)
        output = output.cpu().detach().numpy()
        for i, (e, n) in enumerate(list(zip(output, name))):
            sub.loc[sub['Image'] == n, 'Id'] = ' '.join(lab_encoder.inverse_transform(e.argsort()[-5:][::-1]))

    sub.to_csv('submission.csv', index=False)

def main():
    global args 
    args = arg_parse()
    save_folder = args.save_folder
    weight_decay = args.weight_decay
    momentum = args.momentum
    ngpu = args.ngpu
    iscuda = args.iscuda
    model_name = args.version
    lr = args.lr
    gamma = args.gamma
    bgr_means = cfg['bgr_means']
    img_hw = cfg['img_hw']
    Trainroot = cfg["Trainroot"]
    Valroot = cfg["Valroot"]
    Testroot = cfg["Testroot"]
    epoch_step = cfg['epoch_step']
    start_epoch = cfg['start_epoch']
    end_epoch = cfg['end_epoch']
    pretrained_model = cfg["pretrained_dict"][model_name]
    num_classes = cfg['num_classes']
    #create model and load pretrained weight
    net = model_builder(model_name, pretrained=True, weight_path=pretrained_model, num_classes=num_classes)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    #resume the imcompleted weight into net
    net, start_epoch=resume_checkpoint(net,save_folder,start_epoch)

    if iscuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if ngpu > 1:
        #split the batch data into ngpu
        net = torch.nn.DataParallel(net)
    if iscuda:
        net.cuda()
        #get the best algorithm and configuration
        cudnn.benchmark = True
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    train_transform = TrainTransform(img_hw, bgr_means, padding=False)
    val_transform = ValTransform(img_hw, bgr_means, padding=False)

    # train_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384,384)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop((345,345)),
    #     transforms.RandomRotation((-15,15)),
    #     transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])
    # val_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384,384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])

    #prepare dataset implement
    train_df = pd.read_csv("/home/siting/files/DingGuo/whaledataset/train.csv")
    print(train_df.head())
    train_dff, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    y, lab_encoder = prepare_labels(train_df['Id'])
    # load the dataset
    train_dataset = MyWhaleDataset(Trainroot, 'train', df=train_dff, transform=train_transform, y=y)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    val_dataset = MyWhaleDataset(Valroot, 'val', df=val_df, transform=val_transform, y=y)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=
                                             args.num_workers)

    test_dataset = MyWhaleDataset(Testroot, 'test', transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=
                                             args.num_workers)

    best_prec = 0
    #log the loss
    writer =SummaryWriter(log_dir='./log/test')

    for epoch in range(start_epoch, end_epoch):

        train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, lr, writer)
        prec, loss = eval_net(val_loader, net, criterion)
        prec /= len(val_loader)
        is_best = prec > best_prec
        best_prec = max(best_prec, prec)
        save_checkpoint(net, epoch, is_best, save_folder, model_name)
        print("current accuracy:", prec, "best accuracy:", best_prec,"loss:",loss)

    writer.close()

    last_weight_test_net(test_loader, net, lab_encoder)

if __name__ == '__main__':
    main()

