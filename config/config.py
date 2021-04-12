opt = dict(
    voc_data_dir='/dataset/PASCAL2007/VOC2007/',
    min_size=600,  # image resize
    max_size=1000,  # image resize
    num_workers=8,
    test_num_workers=8,

    # sigma for l1_smooth_loss
    rpn_sigma=3.,
    roi_sigma=1.,

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay=0.0005,
    lr_decay=0.1,  # 1e-3 -> 1e-4
    lr=1e-3,

    # visualization
    env='faster-rcnn',  # visdom env
    port=8097,
    plot_every=40,  # vis every N iter

    # preset
    data='voc',
    pretrained_model='vgg16',

    # training
    epoch=14,

    use_adam=False,  # Use Adam optimizer
    use_chainer=False,  # try match everything as chainer
    use_drop=False,  # use dropout in RoIHead
    # debug
    debug_file='/tmp/debugf',

    test_num=10000,
    # model
    save_path='/checkpoints/fasterrcnn_checkpoint',
    load_path=None
)
