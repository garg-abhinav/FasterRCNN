#  update this configuration file according to your network parameters

opt = dict(
    voc_data_dir='/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/',
    voc_data_test='/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/dataset/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/',
    min_size=600,  # image resize
    max_size=1000,  # image resize
    num_workers=8,
    test_num_workers=8,

    # sigma values for smooth l1 loss
    rpn_sigma=3.,
    roi_sigma=1.,

    # parameters for optimizer
    weight_decay=0.0001,
    lr_decay=0.1,  # 1e-3 -> 1e-4
    lr=1e-3,

    # preset
    data='voc',
    pretrained_model='resnet101',  # change this to resnet101/vgg16 for pretrained model
    epoch=20,

    use_chainer=False,  # try match everything as chainer
    use_drop=False,  # use dropout in RoIHead
    # debug
    debug_file='/tmp/debugf',

    # change train to false while evaluating the network
    train=True,
    test_num=10000,
    # model
    save_path='/home/rcam2/Faster_RCNN_resnet/Faster-R-CNN-master/checkpoints/checkpoint_vgg_3_scales_3_ratios_fourth',
    load_path=None,

    # adjust ratios and scales accoriding to your setting.
    ratios=[0.5, 1, 2],
    anchor_scales=[8, 16, 32],
    feat_stride=16
)
