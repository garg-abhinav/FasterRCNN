from data.voc_dataset import VOC_BBOX_LABELS
import numpy as np
import torch as t
from model.faster_rcnn import *
from torch.utils import data as data_
from tqdm import tqdm
import torch
from data.dataset import Dataset, TestDataset
from config.config import opt
from data.utils import read_image

torch.cuda.set_device(0)
from eval_tool import *


def eval_ap(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()


head = FasterRCNNHead(n_class=len(VOC_BBOX_LABELS) + 1, ratios=opt['ratios'], anchor_scales=opt['anchor_scales'],
                      feat_stride=opt['feat_stride'],  model=opt['pretrained_model'])
tail = FasterRCNNTail(n_class=len(VOC_BBOX_LABELS) + 1, ratios=opt['ratios'], anchor_scales=opt['anchor_scales'],
                      feat_stride=opt['feat_stride'], roi_size=7, model=opt['pretrained_model'])

if torch.cuda.is_available():
    print('CUDA AVAILABLE')
    Faster_RCNN = FasterRCNN(head, tail).cuda()
else:
    print('CUDA NOT AVAILABLE')
    Faster_RCNN = FasterRCNN(head, tail)

dataset = Dataset(opt)
dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt['num_workers'])

testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt['test_num_workers'],
                                   pin_memory=True)

Faster_RCNN.load(opt['save_path'])

opt['train'] = False
results = eval_ap(test_dataloader, Faster_RCNN)

with open("ap.npy", 'wb') as f:
    np.save(f, results['ap'])

with open("map.npy", 'wb') as f:
    np.save(f, results['map'])

img = read_image('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/000092.jpg')
img = t.from_numpy(img)[None]

print(img.shape)

bboxes, labels, scores = Faster_RCNN.predict(img, visualize=True)

with open("img1.npy", 'wb') as f:
    np.save(f, tonumpy(img[0]))

with open("bboxes1.npy", 'wb') as f:
    np.save(f, tonumpy(bboxes[0]))

with open("labels1.npy", 'wb') as f:
    np.save(f, tonumpy(labels[0]))

with open("scores1.npy", 'wb') as f:
    np.save(f, tonumpy(scores[0]))

img = read_image('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/000085.jpg')
img = t.from_numpy(img)[None]

print(img.shape)

bboxes, labels, scores = Faster_RCNN.predict(img, visualize=True)

with open("img2.npy", 'wb') as f:
    np.save(f, tonumpy(img[0]))

with open("bboxes2.npy", 'wb') as f:
    np.save(f, tonumpy(bboxes[0]))

with open("labels2.npy", 'wb') as f:
    np.save(f, tonumpy(labels[0]))

with open("scores2.npy", 'wb') as f:
    np.save(f, tonumpy(scores[0]))

img = read_image('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/000103.jpg')
img = t.from_numpy(img)[None]

print(img.shape)

bboxes, labels, scores = Faster_RCNN.predict(img, visualize=True)

with open("img3.npy", 'wb') as f:
    np.save(f, tonumpy(img[0]))

with open("bboxes3.npy", 'wb') as f:
    np.save(f, tonumpy(bboxes[0]))

with open("labels3.npy", 'wb') as f:
    np.save(f, tonumpy(labels[0]))

with open("scores3.npy", 'wb') as f:
    np.save(f, tonumpy(scores[0]))

img = read_image('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/000185.jpg')
img = t.from_numpy(img)[None]

print(img.shape)

bboxes, labels, scores = Faster_RCNN.predict(img, visualize=True)

with open("img4.npy", 'wb') as f:
    np.save(f, tonumpy(img[0]))

with open("bboxes4.npy", 'wb') as f:
    np.save(f, tonumpy(bboxes[0]))

with open("labels4.npy", 'wb') as f:
    np.save(f, tonumpy(labels[0]))

with open("scores4.npy", 'wb') as f:
    np.save(f, tonumpy(scores[0]))

img = read_image('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/000191.jpg')
img = t.from_numpy(img)[None]

print(img.shape)

bboxes, labels, scores = Faster_RCNN.predict(img, visualize=True)

with open("img5.npy", 'wb') as f:
    np.save(f, tonumpy(img[0]))

with open("bboxes5.npy", 'wb') as f:
    np.save(f, tonumpy(bboxes[0]))

with open("labels5.npy", 'wb') as f:
    np.save(f, tonumpy(labels[0]))

with open("scores5.npy", 'wb') as f:
    np.save(f, tonumpy(scores[0]))

'''        
map_faster = []

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
        rpn_loc_loss_log.append(rpn_loc_loss.cpu().detach().numpy())
        rpn_cls_loss_log.append(rpn_cls_loss.cpu().detach().numpy())
        roi_loc_loss_log.append(roi_loc_loss.cpu().detach().numpy())
        roi_cls_loss_log.append(roi_cls_loss.cpu().detach().numpy())
        total_loss_log.append(total_loss.cpu().detach().numpy())

    opt['train'] = False    
    results = eval_ap(test_dataloader, Faster_RCNN)
    map_faster.append(results['map'])
    opt['train'] = True

    if epoch>=2 and map_faster[epoch] > map_faster[epoch-1]:
        Faster_RCNN.save()
    else:
        Faster_RCNN.save()

    if epoch == 9:
        Faster_RCNN.load('/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/checkpoints/checkpoint_resnet_1_scale_1_ratio_first')
        opt['lr'] = opt['lr'] * opt['lr_decay']

    with open("logs/rpn_loc_loss_res_1_1_first.txt", "w") as f:
        for s in rpn_loc_loss_log:
            f.write(str(s) + '\n')

    with open("logs/rpn_cls_loss_log_res_1_1_first.txt", "w") as f:
        for s in rpn_cls_loss_log:
            f.write(str(s) + '\n')

    with open("logs/roi_loc_loss_log_res_1_1_first.txt", "w") as f:
        for s in roi_loc_loss_log:
            f.write(str(s) + '\n')

    with open("logs/roi_cls_loss_log_res_1_1_first.txt", "w") as f:
        for s in roi_cls_loss_log:
            f.write(str(s) + '\n')

    with open("logs/total_loss_log_res_1_1_first.txt", "w") as f:
        for s in total_loss_log:
            f.write(str(s) + '\n')

    with open("ap_res_1_1_first.npy", 'wb') as f:
        np.save(f, results['ap']) 

    with open("map_res_1_1_first.npy", 'wb') as f:
        np.save(f, results['map'])   

    with open("map_res_1_1_first.txt", "w") as f:
        for s in map_faster:
            f.write(str(s) + '\n')   
'''