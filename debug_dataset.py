from PIL import Image
import numpy as np
import os
from mmseg.datasets.vspw import VSPWDataset, LoadVideoSequenceFromFile
import matplotlib.pyplot as plt
import cv2


def show_one_sample(sample):
    # print(sample.keys())
    # print(sample)
    # print()
    print(sample["inputs"].shape)
    print(sample["data_samples"].gt_sem_seg.data.shape)
    print(sample["data_samples"].metainfo)
    # print(sample["data_samples"])
    # imgs = sample["inputs"]
    # seg_maps = sample["data_samples"].gt_sem_seg.data
    # for img, seg_map in zip(imgs, seg_maps):
    #     cv2.imshow("Image", img.permute(1,2,0).numpy())
    #     cv2.imshow("Segmentation", seg_map.numpy())
    #     cv2.waitKey(0)

train_dataset = VSPWDataset(data_root="./VSPW_480p", ann_file="train.txt", seq_length=5)
train_pipeline = [dict(type="LoadVideoSequenceFromFile", resize=(640, 480))]

val_dataset = VSPWDataset(data_root="./VSPW_480p", ann_file="val.txt", seq_length=5)
val_pipeline = [dict(type="LoadVideoSequenceFromFile", resize=(640, 480))]


# test pipline
loader = LoadVideoSequenceFromFile(flip_prob=0.5)

# print(len(train_dataset))
# print(len(val_dataset))
# visualize data
for sample in train_dataset:
    # load sample via pipeline
    sample = loader(sample)

    show_one_sample(sample)
    print(sample["is_flipped"])
s = 0