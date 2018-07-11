
# coding: utf-8

import os
import glob
import json
import random
import time
import SimpleITK as sitk
import numpy as np
from msd_amc_preprocess import preprocess
from msd_amc_model.model import average_dice_coef, average_dice_coef_loss, load_model, data_aug
from tensorflow.python.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"]="0" ##################### modify
task = 'Task01_BrainTumour' ##################### modify
#tasks = ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate','Task06_Lung','Task07_Pancreas']

path = "/data/public/rw/medical/decathlon2018_rev0704"
with open(os.path.join(path, task, 'set1.json')) as f:
    cases = json.load(f)
train_cases = [os.path.join(path, task, case['image'][2:]) for case in cases['training']]
valid_cases = [os.path.join(path, task, case['image'][2:]) for case in cases['validation']]
print('# of training data : ', len(train_cases), ', # of validation data : ', len(valid_cases))

config_data = preprocess.preprocess(path, task)
input_shape = (config_data.img_dep, config_data.img_rows, config_data.img_cols, config_data.num_channels)
num_labels = config_data.num_labels
num_channels = config_data.num_channels
train_mean_sd = config_data.train_mean_sd
model = load_model(input_shape=input_shape,
                    num_labels=num_labels,
                    base_filter=32,
                    se_block=True,
                    se_ratio=16)
model.compile(optimizer=Adam(lr=1e-4), loss=average_dice_coef_loss, metrics=[average_dice_coef])
datagen = data_aug(task)

score = [0.]
num_steps = 10000
for n in range(num_steps):
    random.shuffle(train_cases)
    start_time = time.time()
    
    train_dice = 0
    for case in train_cases:
        img = sitk.GetArrayFromImage(sitk.ReadImage(case))
        img = config_data.preprocess_img(img)
        for channel in range(num_channels):
            img[...,channel] -= train_mean_sd[channel][0]
            img[...,channel] /= train_mean_sd[channel][1]
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(case.replace('imagesTr', 'labelsTr')))
        mask = config_data.hierarchical_preprocess_mask(mask)
        
        img = config_data.resize_3d(img, num_channels)
        mask = config_data.resize_3d(mask, num_labels)
        
        img = np.expand_dims(img,0)
        mask = np.expand_dims(mask,0)
        seed = np.random.randint(0, 100, 1)[0]
        for img in datagen.flow(img, batch_size=1, seed = seed):
            break
        for mask in datagen.flow(mask, batch_size=1, seed = seed):
            break
        if len(mask[mask == 1]) == 0:
            raise Exception('mask is empty', case)
        train_dice += model.train_on_batch(img, mask)[-1]
    train_dice /= len(train_cases)
    
    valid_dice = 0
    for case in valid_cases:
        img = sitk.GetArrayFromImage(sitk.ReadImage(case))
        img = config_data.preprocess_img(img)
        for channel in range(num_channels):
            img[...,channel] -= train_mean_sd[channel][0]
            img[...,channel] /= train_mean_sd[channel][1]
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(case.replace('imagesTr', 'labelsTr')))
        mask = config_data.hierarchical_preprocess_mask(mask)
        
        img = config_data.resize_3d(img, num_channels)
        mask = config_data.resize_3d(mask, num_labels)
        
        img = np.expand_dims(img,0)
        mask = np.expand_dims(mask,0)
        valid_dice += model.evaluate(img, mask, verbose=0)[-1]
    valid_dice /= len(valid_cases)

    end_time = time.time()
    print("Wall time : %.2f seconds" % (end_time - start_time), train_dice, valid_dice)

    if np.max(score) <= valid_dice:
        model.save_weights('./180711_{}_baseline_final.h5'.format(task.replace('Task','').replace('_', '')))

    score.append(valid_dice)
    try:  
        f = open("./180711_{}_baseline_final.txt".format(task.replace('Task','').replace('_', '')), 'a')
    except:
        print("ERROR: can't open file")
        sys.exit(1)
    f.write("train_dice:{}/test_dice:{}\n".format(train_dice,valid_dice))
    f.close()

