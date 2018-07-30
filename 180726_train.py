# coding: utf-8

import os
import glob
import json
import random
import time
import SimpleITK as sitk
import numpy as np
from msd_amc_preprocess import preprocess
from msd_amc_model.model import average_dice_coef, average_dice_coef_loss, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow import PrefetchData
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("visible_gpu_num", type=str, help="gpu number")
#parser.add_argument("data_path", type=str)
parser.add_argument("task_num", type=int, help="1 ~ 7")
parser.add_argument("patch_whole", type=str, help="select 'patch' or 'whole'")
parser.add_argument("split_index", type=int, help="0 ~ 3 for 4fold-CV")
args = parser.parse_args()

if args.task_num < 1 or args.split_index > 7:
    raise TaskInputError('Task number is 1 ~ 7, but got {}.'.format(args.task_num))
if args.patch_whole != 'whole' and args.patch_whole != 'patch':
    raise ModelError('select "patch" or "whole" model, got {}'.format(args.patch_whole))
if args.split_index < 0 or args.split_index > 3:
    raise DataSplitError('0 ~ 3, 4 fold-CV is defined, but got {}'.format(args.split_index))

print('model : ', args.patch_whole)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu_num
print('gpu : ',args.visible_gpu_num)
path = "/data/private/beomheep/00.MSD/decathlon2018_1mm"
tasks = ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate','Task06_Lung','Task07_Pancreas']
task = tasks[args.task_num-1]
print('task : ', task)
train_cases = glob.glob(os.path.join(path, task, 'imagesTr', '*'))
valid_cases = train_cases[args.split_index::4].copy()
print('{} CV'.format(args.split_index))
for i in valid_cases:
    train_cases.remove(i)
print('# of training data : ', len(train_cases), ', # of validation data : ', len(valid_cases))

config_task = preprocess.Preprocess(path, task)
input_shape = (None, None, None, config_task.num_channels)
if args.patch_whole == 'whole':
    num_labels = 1
else:
    num_labels = config_task.num_labels
num_channels = config_task.num_channels
model = load_model(input_shape=input_shape,
                    num_labels=num_labels,
                    base_filter=32,
                    depth_size=3,
                    se_res_block=True,
                    se_ratio=16,
                    last_relu=True)
model.compile(optimizer=Adam(lr=1e-4), loss=average_dice_coef_loss, metrics=[average_dice_coef])

ds_train = preprocess.DataLoader(train_cases, is_shuffle=True)
ds_train = MapDataComponent(ds_train, config_task.preprocess_img, index=0)
ds_train = MapDataComponent(ds_train, config_task.sample_z_norm, index=0)
ds_train = MapDataComponent(ds_train, config_task.simple_preprocess_mask, index=1)
if args.patch_whole == 'whole':
    ds_train = MapDataComponent(ds_train, config_task.whole_merge_mask, index=1)
    ds_train = MapData(ds_train, config_task.resize_whole)
else:
    ds_train = MapData(ds_train, config_task.crop_to_patch)
    ds_train = MapData(ds_train, config_task.resize_patch)
ds_train = MapData(ds_train, config_task.data_aug)
ds_train = PrefetchData(ds_train, 2, 1)

ds_valid = preprocess.DataLoader(valid_cases, is_shuffle=False)
ds_valid = MapDataComponent(ds_valid, config_task.preprocess_img, index=0)
ds_valid = MapDataComponent(ds_valid, config_task.sample_z_norm, index=0)
ds_valid = MapDataComponent(ds_valid, config_task.simple_preprocess_mask, index=1)
if args.patch_whole == 'whole':
    ds_valid = MapDataComponent(ds_valid, config_task.whole_merge_mask, index=1)
    ds_valid = MapData(ds_valid, config_task.resize_whole)
else:
    ds_valid = MapData(ds_valid, config_task.crop_to_patch)
    ds_valid = MapData(ds_valid, config_task.resize_patch)
ds_valid = PrefetchData(ds_valid, 2, 1)

gen_train = preprocess.gen_data(ds_train)
gen_valid = preprocess.gen_data(ds_valid)

cbs = list()
cbs.append(ModelCheckpoint('./180726_{}_{}_{}.h5'.format(task,args.patch_whole,args.split_index), save_best_only=True, save_weights_only=True))
cbs.append(CSVLogger('./180726_{}_{}_{}.log'.format(task,args.patch_whole,args.split_index), append=True))

model.fit_generator(generator=gen_train,
                    steps_per_epoch=len(train_cases),
                    epochs=1000,
                    validation_data=gen_valid,
                    validation_steps=len(valid_cases),
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=cbs)
