
# coding: utf-8

import os
import glob
import numpy as np
from scipy import ndimage
import json

class preprocess(object):
    def __init__(self, path, task):
        self.path = path
        self.task = task

        with open(os.path.join(self.path, self.task, 'dataset.json')) as f:
            self.data = json.load(f)
    
        self.num_labels = len(self.data['labels'])-1
        self.num_channels = len(self.data['modality'])

        if self.task == 'Task01_BrainTumour':
            self.img_dep, self.img_rows, self.img_cols = 155, 240, 240
            self.train_mean_sd = [[0.049055588, 0.11951996],[0.060917288, 0.14406842],[0.042638578, 0.10113136],[0.052024964, 0.1285218]]
        elif self.task == 'Task02_Heart':
            self.img_dep, self.img_rows, self.img_cols = None, None, None
            self.train_mean_sd = [[0.08674451543225183, 0.13093200988239712]]
        elif self.task == 'Task03_Liver':
            self.img_dep, self.img_rows, self.img_cols = 160, 256, 256
            self.train_mean_sd = [[0.12085412, 0.12076765]]
        elif self.task == 'Task04_Hippocampus':
            self.img_dep, self.img_rows, self.img_cols = None, None, None
            self.train_mean_sd = [[0.28123286390750346, 0.12707186294415773]]
        elif self.task == 'Task05_Prostate':
            self.img_dep, self.img_rows, self.img_cols = 32, 320, 320
            self.train_mean_sd = [[0.17949264, 0.14028978],[0.12471767, 0.16693035]]
        elif self.task == 'Task06_Lung':
            self.img_dep, self.img_rows, self.img_cols = 160, 256, 256
            self.train_mean_sd = [[0.10726791, 0.11925137]]
        elif self.task == 'Task07_Pancreas':
            self.img_dep, self.img_rows, self.img_cols = 112, 304, 304
            self.train_mean_sd = [[0.110755935, 0.11887076]]
        else:
            raise Exception(self.task, 'task is not defined.')

    def preprocess_img(self, img):
        if self.data['modality']['0'] == 'CT':
            low = -1024.
            high = 3071.
        else:
            low = np.min(img)
            high = np.max(img)
            
        if len(self.data['modality']) == 1: 
            img = (img-low) / (high-low)
            img[img < 0.] = 0
            img[img > 1.] = 1
            img = np.expand_dims(img, -1)
        else:
            img = img.transpose((1,2,3,0)) # channels last
            for self.num_channel in range(self.num_channels):
                img[...,self.num_channel] = (img[...,self.num_channel]-low) / (high-low)
                img[...,self.num_channel][img[...,self.num_channel] < 0.] = 0
                img[...,self.num_channel][img[...,self.num_channel] > 1.] = 1
        return img

    def simple_preprocess_mask(self, mask):
        mask_dummy = np.zeros(mask.shape+(self.num_labels,))
        for self.num_label in range(self.num_labels): # except background
            mask_dummy[:,:,:,self.num_label][mask == self.num_label+1] = 1.
        return mask_dummy

    def hierarchical_preprocess_mask(self, mask):
        mask_dummy = np.zeros(mask.shape+(self.num_labels,))
        if self.task == 'Task01_BrainTumour' or self.task == 'Task04_Hippocampus':
            for self.num_label in range(self.num_labels): # except background
                mask_dummy[:,:,:,self.num_label][mask == self.num_label+1] = 1.
        else:
            for self.num_label in range(self.num_labels):
                for re_num_label in reversed(range(self.num_labels)):
                    mask_dummy[:,:,:,self.num_label][mask == re_num_label+1] = 1
                    if re_num_label == self.num_label:
                        break
        return mask_dummy

    def resize_3d(self, img, num_last_axis):
        if self.task == 'Task01_BrainTumour':
            resize_factor = (155./img.shape[0], 240./img.shape[1], 240./img.shape[2])
        elif self.task == 'Task02_Heart':
            resize_factor = (0.9, 0.9, 0.9)
        elif self.task == 'Task03_Liver':
            resize_factor = (160./img.shape[0],256./img.shape[1],256./img.shape[2])
        elif self.task == 'Task04_Hippocampus':
            resize_factor = (1., 1., 1.)
        elif self.task == 'Task05_Prostate':
            resize_factor = (32./img.shape[0],320./img.shape[1],320./img.shape[2])
        elif self.task == 'Task06_Lung':
            resize_factor = (160./img.shape[0],256./img.shape[1],256./img.shape[2])
        elif self.task == 'Task07_Pancreas':
            resize_factor = (112./img.shape[0],304./img.shape[1],304./img.shape[2])
        else:
            raise Exception(self.task, 'task is not defined.')
            
        resized_img = []
        for num_last in range(num_last_axis):
            resized_img_dummy = ndimage.zoom(img[...,num_last], resize_factor, order=0, mode='constant', cval=0.0)
            resized_img.append(resized_img_dummy)
        resized_img = np.array(resized_img).transpose((1,2,3,0))
        return resized_img

    def resize_to_origin(self, img, origin_size):
        resize_factor = (origin_size[0]/img.shape[0], origin_size[1]/img.shape[1], origin_size[2]/img.shape[2])
        resized_img = []
        for num_label in range(self.num_labels):            
            resized_img_dummy = ndimage.zoom(img[...,self.num_label], resize_factor, order=0, mode='constant', cval=0.0)
            resized_img.append(resized_img_dummy)
        resized_img = np.array(resized_img).transpose((1,2,3,0))
        return resized_img