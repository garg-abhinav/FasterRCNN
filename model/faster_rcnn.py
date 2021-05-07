import torch as t
import numpy as np
from .utils.bbox_tools import loc2bbox
from torchvision.ops import nms
import time
import os
from .region_proposal_network import RegionProposalNetwork
from .utils.creator_tool import ProposalTargetCreator, AnchorTargetCreator
from data.utils import preprocess
from config.config import opt
from torchvision.models import vgg16, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.ops import RoIPool

from torch import nn
from torch.nn import functional as F


def nograd(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):
    def __init__(self, faster_rcnn_head, faster_rcnn_tail):

        super(FasterRCNN, self).__init__()
        self.head = faster_rcnn_head
        self.tail = faster_rcnn_tail
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.optimizer = self.get_optimizer()
        self.n_class = self.head.n_class
        self.rpn_sigma = opt['rpn_sigma']
        self.roi_sigma = opt['roi_sigma']
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    def forward(self, x, scale=1.0):

        features, rpn_locs, rpn_scores, rois, roi_indices, anchor = self.head(x, scale=1.)
        roi_cls_locs, roi_scores = self.tail(features, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def train_rpn_batch(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]

        if n != 1:
            raise ValueError('Only batch size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features, rpn_locs, rpn_scores, rois, roi_indices, anchor = self.head(imgs, scale=1.)

        # since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # --------------RPN losses--------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox), anchor, img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)

        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        ### don't know what this does
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        ###

        sum = rpn_loc_loss + rpn_cls_loss

        self.optimizer.zero_grad()
        sum.backward()
        self.optimizer.step()

        return rpn_loc_loss, rpn_cls_loss, sum

    def train_rcnn_batch(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]

        if n != 1:
            raise ValueError('Only batch size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features, rpn_locs, rpn_scores, rois, roi_indices, anchor = self.head(imgs, scale=1.)

        # since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = t.zeros(len(sample_roi))

        roi_cls_loc, roi_score = self.tail(features, sample_roi, sample_roi_index)

        # --------------RPN losses--------------#
        # gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox), anchor, img_size)
        # gt_rpn_label = totensor(gt_rpn_label).long()
        # gt_rpn_loc = totensor(gt_rpn_loc)

        # rpn_loc_loss = _fast_rcnn_loc_loss(
        #     rpn_loc,
        #     gt_rpn_loc,
        #     gt_rpn_label.data,
        #     self.rpn_sigma)
        #
        # rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        ### don't know what this does
        # _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        # _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        ###

        # -----------ROI losses-------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        sum = roi_loc_loss + roi_cls_loss

        self.optimizer.zero_grad()
        sum.backward()
        self.optimizer.step()

        return roi_loc_loss, roi_cls_loss, sum

    def train_batch(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]

        if n != 1:
            raise ValueError('Only batch size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features, rpn_locs, rpn_scores, rois, roi_indices, anchor = self.head(imgs, scale=1.)

        # since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = t.zeros(len(sample_roi))

        roi_cls_loc, roi_score = self.tail(features, sample_roi, sample_roi_index)

        # --------------RPN losses--------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox), anchor, img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)

        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        ### don't know what this does
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        ###

        # -----------ROI losses-------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        sum = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        self.optimizer.zero_grad()
        sum.backward()
        self.optimizer.step()

        return rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, sum

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': opt['lr'] * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': opt['lr'], 'weight_decay': opt['weight_decay']}]

        self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def save(self):
        save_dict = dict()

        save_dict['head'] = self.head.state_dict()
        save_dict['tail'] = self.tail.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        save_path = opt['save_path']

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)

    def load(self, path):
        state_dict = t.load(path)
        if 'head' in state_dict:
            self.head.load_state_dict(state_dict['head'])
        if 'tail' in state_dict:
            self.tail.load_state_dict(state_dict['tail'])
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def use_preset(self, preset):
        """Use the given preset during prediction.
        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.
        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.
        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):

        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self.forward(img, scale=scale)

            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),
                                tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        return bboxes, labels, scores


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class FasterRCNNHead(nn.Module):
    def __init__(self, n_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16, model='vgg16'):
        super(FasterRCNNHead, self).__init__()
        if model == 'vgg16':
            pretrained = vgg16(pretrained=True)
            feature_extractor = list(pretrained.features)[:30]

            for layer in feature_extractor[:10]:
                for p in layer.parameters():
                    p.requires_grad = False

            # print(feature_extractor)
            self.feature_extractor = nn.Sequential(*feature_extractor)
        elif model == 'resnet101':
            pretrained = resnet101(pretrained=True)
            self.feature_extractor = nn.Sequential(pretrained.conv1, pretrained.bn1, pretrained.relu,
                                          pretrained.maxpool, pretrained.layer1, pretrained.layer2, pretrained.layer3)

            for i in [0, 1, 4, 5, 6]:
                for p in self.feature_extractor[i].parameters():
                    p.requires_grad = False

            self.feature_extractor.apply(set_bn_fix)
        else:
            raise ValueError(f'Model {model} has not been implemented yet')

        self.rpn = RegionProposalNetwork(512, 512, ratios=ratios, anchor_scales=anchor_scales, feat_stride=feat_stride)

        self.n_class = n_class
        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride

    def forward(self, x, scale=1.):  # x input tensor
        img_size = x.shape[2:]

        features = self.feature_extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(features, img_size, scale)

        return features, rpn_locs, rpn_scores, rois, roi_indices, anchor


class FasterRCNNTail(nn.Module):
    def __init__(self, n_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16, roi_size=7,
                 model='vgg16'):
        super(FasterRCNNTail, self).__init__()
        self.n_class = n_class
        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride
        self.spatial_scale = 1.0 / feat_stride
        self.roi_size = roi_size
        self.base_model = model

        if self.base_model == 'vgg16':
            pretrained = vgg16(pretrained=True)
            classifier = pretrained.classifier
            classifier = list(classifier)
            del classifier[6]
            del classifier[5]
            del classifier[2]
            self.classifier = nn.Sequential(*classifier)

        elif self.base_model == 'resnet101':
            pretrained = resnet101(pretrained=True)
            self.classifier = nn.Sequential(pretrained.layer4)

        else:
            raise ValueError(f'Model {self.base_model} has not been implemented yet')

        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, features, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(features, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        if self.base_model == 'resnet101':
            fc7 = self.classifier(pool).mean(3).mean(2)
        else:
            fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False
