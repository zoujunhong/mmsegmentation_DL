# dataset settings
dataset_type = 'VSPWDataset'
data_root = './data/VSPW_480p'
crop_size = (480, 640)
train_pipeline = [dict(type="LoadVideoSequenceFromFile", resize=(640, 480))]
test_pipeline = [dict(type="LoadVideoSequenceFromFile", resize=(640, 480))]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train.txt",
        seq_length=1, # seq_length -> regarded as img
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val.txt",
        seq_length=1, # seq_length -> regarded as img
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
