# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1000, 480), (1000, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='VOCDataset',
            ann_file='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
            img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/',
            pipeline=train_pipeline)),
    #### Evaluate with VOC metric, metric="mAP"; or use toos/semi_dist_test.sh
    val=dict(
        type='VOCDataset',
        ann_file='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type='VOCDataset',
        ann_file='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/',
        pipeline=test_pipeline)
    ##### Evaluate with COCO metric, metric="bbox";or use tools/semi_dist_test.sh
    #val=dict(
    #    type='Voc2CocoDataset',
    #    ann_file='data_list/voc_semi/voc07_test.json',
    #    img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/JPEGImages/',
    #    pipeline=test_pipeline),
    #test=dict(
    #    type='Voc2CocoDataset',
    #    ann_file='data_list/voc_semi/voc07_test.json',
    #    img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2007/JPEGImages/',
    #    pipeline=test_pipeline)

    ##### For inferencing pseudo-labels of the unlabel images via tools/inference_unlabeled_coco_data.sh
    #test=dict(
    #    type='Voc2CocoDataset',
    #    ann_file='data_list/voc_semi/voc12_trainval.json',
    #    img_prefix='/gruntdata2/tcguo/voc/VOCdevkit/VOC2012/JPEGImages/',
    #    pipeline=test_pipeline)
)

# learning policy
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    #_delete_=True, 
    grad_clip=dict(max_norm=10, norm_type=2))
    #grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[50, 60])
runner = dict(type='EpochBasedRunner', max_epochs=60)
### VOC metric use "mAP", COCO metric use "bbox"
evaluation = dict(interval=5, metric='mAP')

checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
