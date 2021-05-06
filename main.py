from model.faster_rcnn_vgg import FasterRCNNHead, FasterRCNNTail, FasterRCNN
from torch.utils import data as data_
from tqdm import tqdm
import torch
from data.dataset import Dataset, TestDataset, ValDataset
from config.config import opt
import numpy as np
import torch as t


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()


head = FasterRCNNHead(n_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16,
                      model=opt['pretrained_model'])
tail = FasterRCNNTail(n_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16, roi_size=7,
                      model=opt['pretrained_model'])

if torch.cuda.is_available():
    print('CUDA AVAILABLE')
    Faster_RCNN = FasterRCNN(head, tail).cuda()
else:
    print('CUDA NOT AVAILABLE')
    Faster_RCNN = FasterRCNN(head, tail)

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
total_loss_log = []

for epoch in range(opt['epoch']):
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        scale = scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss = Faster_RCNN.train_batch(img, bbox, label,
                                                                                                     scale)
        rpn_loc_loss_log.append(rpn_loc_loss)
        rpn_cls_loss_log.append(rpn_cls_loss)
        roi_loc_loss_log.append(roi_loc_loss)
        roi_cls_loss_log.append(roi_cls_loss)
        total_loss_log.append(total_loss)

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

with open("logs/total_loss_log.txt", "w") as f:
    for s in total_loss_log:
        f.write(str(s) + '\n')
