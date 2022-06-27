model = dict(
    type='FCOS',
    backbone=dict(
        type='RLA_ResNet',
        layers=[3,4,6,3],
        #depth=50,
        #num_stages=4,
        #out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='/home/ashui.cbh/.cache/torch/checkpoints/resnet50_rla_2283.pth.tar'),
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
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        # partially data use 3.0; full data use 1.0
        loss_weight = 3.0,
        soft_weight = 1.0,
        soft_warm_up = 5000,
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
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='PatchShuffle', ratio=0.5, ranges=[0.0,1.0], mode=['flip','flop']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg', 'PS', 'PS_place', 'PS_mode']),
]
unlabel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='PatchShuffle', ratio=0.5, ranges=[0.0,1.0], mode=['flip','flop']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAugmentBBox_Fast', aug_type='affine'),
    dict(type='UBAug'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], meta_keys=['filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'scale_idx', 'flip','flip_direction', 'img_norm_cfg', 'PS', 'PS_place', 'PS_mode']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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

dataset_type = 'SemiCOCODataset'
## DSL-style data root; recommend to use absolute path
data_root = '/gruntdata1/bhchen/factory/data/semicoco/'

data = dict(
    samples_per_gpu=2,
    # if you change workers_per_gpu, please change preload below at the same time
    workers_per_gpu=2,
    # batch_config controls the ratio of labeled to unlabeled images within one databatch
    batch_config=dict(ratio =[[1, 1],]),
    train=dict(
        type=dataset_type,
        ann_file = 'data_list/coco_semi/semi_supervised/instances_train2017.2@10.json',
        ann_path = data_root + 'prepared_annos/Industry/annotations/full/',
        labelmapper = data_root + 'mmdet_category_info.json',
        img_prefix = data_root + 'images/full/',
        pipeline = train_pipeline,
        ),
    unlabel_train=dict(
        type=dataset_type,
        ann_file = 'data_list/coco_semi/semi_supervised/instances_train2017.2@10-unlabeled.json',
        ann_path = data_root + 'unlabel_prepared_annos/Industry/annotations/full/',
        labelmapper = data_root + 'mmdet_category_info.json',
        img_prefix = data_root + 'images/full/',
        pipeline = unlabel_train_pipeline,
        # fixed thres like [0.1, 0.4]; or ada thres, this file will be generated during running (please remove it after each training)
        thres="adathres.json",
        ),
    unlabel_pred=dict(
        type=dataset_type,
        num_gpus = 8,
        image_root_path = data_root + "images/full/",
        image_list_file = 'data_list/coco_semi/semi_supervised/instances_train2017.2@10-unlabeled.json',
        anno_root_path = data_root + "unlabel_prepared_annos/Industry/annotations/full/",
        category_info_path = data_root + 'mmdet_category_info.json',
        infer_score_thre=0.1,
        save_file_format="json",
        pipeline = test_pipeline,
        eval_config ={"iou":[0.6]},
        img_path = data_root + "images/full/",
        img_resize_size = (1333,800),
        low_level_scale = 16,
        use_ema=True,
        eval_flip=False,
        fuse_history=False, # as ISMT fuse history bboxes by nms
        first_fuse=False,
        first_score_thre=0.1,
        eval_checkpoint_config=dict(interval=1, mode="iteration"),
        # 2*num_worker+2
        preload=6,
        #the start epoch; partial data use 8; alldata don't use by setting 100
        start_point=8),
    val=dict(
        type = 'CocoDataset',
        ann_file = 'data_list/coco_semi/semi_supervised/instances_val2017.json',
        img_prefix = data_root + 'valid_images/full/',
        pipeline = test_pipeline,
        ),
    test=dict(
        type = 'CocoDataset',
        ann_file = 'data_list/coco_semi/semi_supervised/instances_val2017.json',
        img_prefix = data_root + 'valid_images/full/',
        pipeline = test_pipeline,
        )
)
# learning policy
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    #_delete_=True, 
    grad_clip=dict(max_norm=35, norm_type=2))
    #grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    # partial data use 20-26-28; full data use 20-32-34
    step=[20, 26])
runner = dict(type='SemiEpochBasedRunner', max_epochs=28)
evaluation = dict(interval=1, metric='bbox')

checkpoint_config = dict(interval=1)
ema_config = dict(interval=1, mode="iteration",ratio=0.99,start_point=1)
scale_invariant = True

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
