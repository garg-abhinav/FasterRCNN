from model.faster_rcnn import FasterRCNNHead, FasterRCNNTail, FasterRCNN
from torch.utils import data as data_
from tqdm import tqdm
import torch
from data.dataset import Dataset, TestDataset, ValDataset
from data.voc_dataset import VOC_BBOX_LABELS
from config.config import opt
import numpy as np
import torch as t


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()


head = FasterRCNNHead(n_class=len(VOC_BBOX_LABELS)+1, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16,
                      model=opt['pretrained_model'])
tail = FasterRCNNTail(n_class=len(VOC_BBOX_LABELS)+1, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16,
                      roi_size=7, model=opt['pretrained_model'])


'''
This code was written for alternating training strategy; however, we couldn't explore this strategy due to the time
constraint.
'''

if torch.cuda.is_available():
    print('CUDA AVAILABLE')
    faster_rcnn = FasterRCNN(head, tail).cuda()
else:
    print('CUDA NOT AVAILABLE')
    faster_rcnn = FasterRCNN(head, tail)

dataset = Dataset(opt)
dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt['num_workers'])

valset = ValDataset(opt)
val_dataloader = data_.DataLoader(valset, batch_size=1, shuffle=False, num_workers=opt['num_workers'],
                                  pin_memory=True)

testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt['test_num_workers'],
                                   pin_memory=True)

rpn_loc_loss_log = []
rpn_cls_loss_log = []
roi_loc_loss_log = []
roi_cls_loss_log = []
total_rpn_loss_log = []
total_rcnn_loss_log = []

for iteration in range(opt['epoch']):
    # Freeze Tail Layers
    for layer in faster_rcnn.tail.children():
        for p in layer.parameters():
            p.requires_grad = False

    # RPN Training
    for epoch in range(opt['epoch']):
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            rpn_loc_loss, rpn_cls_loss, total_rpn_loss = faster_rcnn.train_rpn_batch(img, bbox, label, scale)
            rpn_loc_loss_log.append(rpn_loc_loss)
            rpn_cls_loss_log.append(rpn_cls_loss)
            total_rpn_loss_log.append(total_rpn_loss)

    # Unfreeze Tail Layers
    for layer in faster_rcnn.tail.children():
        for p in layer.parameters():
            p.requires_grad = True

    # Freeze Head Layers
    for layer in faster_rcnn.head.children():
        for p in layer.parameters():
            p.requires_grad = False

    # Fast R-CNN Training
    for epoch in range(opt['epoch']):
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            roi_loc_loss, roi_cls_loss, total_rcnn_loss = faster_rcnn.train_rcnn_batch(img, bbox, label, scale)
            roi_loc_loss_log.append(roi_loc_loss)
            roi_cls_loss_log.append(roi_cls_loss)
            total_rcnn_loss_log.append(total_rcnn_loss)

    # TODO: how to use rcnn tuned network to initialize rpn in the next iteration?

    # Unfreeze head layers
    for layer in faster_rcnn.head.children():
        for p in layer.parameters():
            p.requires_grad = True

    # Freeze the first few pretrained model layers
    for layer in faster_rcnn.head.feature_extractor[:10].children():
        for p in layer.parameters():
            p.requires_grad = False

with open("logs/rpn_loc_loss.txt", "w") as f:
    for s in rpn_loc_loss_log:
        f.write(str(s) + '\n')

with open("logs/rpn_cls_loss_log.txt", "w") as f:
    for s in rpn_cls_loss_log:
        f.write(str(s) + '\n')

with open("logs/roi_loc_loss_log.txt", "w") as f:
    for s in roi_loc_loss_log:
        f.write(str(s) + '\n')

with open("logs/roi_cls_loss_log.txt", "w") as f:
    for s in roi_cls_loss_log:
        f.write(str(s) + '\n')

with open("logs/total_rpn_loss_log.txt", "w") as f:
    for s in total_rpn_loss_log:
        f.write(str(s) + '\n')

with open("logs/total_rcnn_loss_log.txt", "w") as f:
    for s in total_rcnn_loss_log:
        f.write(str(s) + '\n')
