from PIL import Image
import numpy as np
import os
from mmseg.datasets.vspw import VSPWDataset, LoadVideoSequenceFromFile

train_dataset = VSPWDataset(data_root="./VSPW_480p", ann_file="train.txt", seq_length=5)
train_pipline = [dict(type="LoadVideoSequenceFromFile")]

val_dataset = VSPWDataset(data_root="./VSPW_480p", ann_file="val.txt", seq_length=5)
val_pipline = [dict(type="LoadVideoSequenceFromFile")]

# load one sample
sample = train_dataset[0]

# test pipline
loader = LoadVideoSequenceFromFile()
sample_ = loader(sample)