import os


pretrained_dict = {
    "vgg16"    : "/home/siting/.torch/models/vgg16_reducedfc.pth",
    "resnet50" : "/home/siting/.torch/models/resnet50-19c8e357.pth",
    "resnet101" : "/home/siting/.torch/models/resnet101-5d3b4d8f.pth",
    "resnet152" : "/home/siting/.torch/models/resnet152-b121ed2d.pth",
    "densenet121" : "/home/siting/.torch/models/densenet121-a639ec97.pth",
    "densenet161" : "/home/siting/.torch/models/densenet161-17b70270.pth",
    "senet154" : "/home/siting/.torch/models/senet154-c7b49a05.pth",
    "se_resnet101" : "/home/siting/.torch/models/se_resnet101-7e38fcc6.pth",
    "inceptionv4" : "/home/siting/.torch/models/inceptionv4-8e4777a0.pth",
    "se_resnext101_32x4d" : "/home/siting/.torch/models/se_resnext101_32x4d-3b2fe3d8.pth",
    "xception" : "/home/siting/.torch/models/xception-b0b7af25.pth",
    "inceptionresnetv2" : "/home/siting/.torch/models/inceptionresnetv2-520b38e4.pth"
}


mydataset = {
    'Trainroot' : "/home/siting/files/DingGuo/whaledataset/",
    'Valroot' : "/home/siting/files/DingGuo/whaledataset/",
    'pretrained_dict' : pretrained_dict,
    'bgr_means' : (104, 117, 123),
    'img_hw' : (384, 384),
    'start_epoch' : 0,
    'end_epoch' : 50,
    'epoch_step' : [0, 9, 12],
    'save_folder' : './weights/',
    'num_classes' : 5005,
    'Testroot' : "/home/siting/files/DingGuo/whaledataset/"
}

voc = {
    'Trainroot' : "./data/datasets/voc/set/",
    'Valroot' : "./data/datasets/voc/set/",
    'pretrained_dict' : pretrained_dict,
    'bgr_means' : (104, 117, 123),
    'img_hw' : (300, 300),
    'start_epoch' : 0,
    'end_epoch' : 16,
    'epoch_step' : [0, 9, 14],
    'save_folder' : './weights/',
    'num_classes' : 20,
    'Testroot' : "./data/datasets/voc/set/"
}