# dataset settings
dataset_type = 'VPSDataset'
data_root = 'data/ivps/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (448, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_ivps', reduce_zero_label=False),
    dict(type='Resize', img_scale=(448, 256), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=0),   ##changed by guolei
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root+'VPS-TrainSet',
            img_dir='',
            ann_dir='',
            split='train',
            pipeline=train_pipeline,
            video_dataset_list=["ASU-Mayo_Clinic", "CVC-ClinicDB-612", "CVC-ColonDB-300"],
            dilation=[-3,-2,-1])),
    val=dict(
        type=dataset_type,
        data_root=data_root+'VPS-TestSet',
        img_dir='',
        ann_dir='',
        split='val',
        pipeline=test_pipeline,
        video_dataset_list=["CVC-ClinicDB-612-Test", "CVC-ClinicDB-612-Valid", "CVC-ColonDB-300"],
        dilation=[-3,-2,-1]),
    test=dict(
        type=dataset_type,
        data_root=data_root+'VPS-TestSet',
        img_dir='',
        ann_dir='',
        split='val',
        pipeline=test_pipeline,
        video_dataset_list=["CVC-ClinicDB-612-Test", "CVC-ClinicDB-612-Valid", "CVC-ColonDB-300"],
        dilation=[-3,-2,-1]))
