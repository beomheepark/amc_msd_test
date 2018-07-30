
# coding: utf-8

import os
import glob
import numpy as np
from scipy import ndimage
import json
import SimpleITK as sitk
from skimage import exposure
import math
import random
from msd_amc_preprocess import keras_image3d
from tensorpack.dataflow.base import RNGDataFlow

class DataLoader(RNGDataFlow):
    def __init__(self, file_paths, is_shuffle=True):
        self.file_paths = file_paths
        self.is_shuffle = is_shuffle

    def get_data(self):
        if self.is_shuffle == True:
            random.shuffle(self.file_paths)
        for file_path in self.file_paths:
            image_dummy = sitk.ReadImage(file_path)
            image = sitk.GetArrayFromImage(image_dummy).astype('float32')
            resize_factor = image_dummy.GetSpacing()[::-1]
            image = ndimage.zoom(image, resize_factor, order=0, mode='constant', cval=0.0)

            label_dummy = sitk.ReadImage(file_path.replace('imagesTr', 'labelsTr'))
            label = sitk.GetArrayFromImage(label_dummy).astype('float32')
            resize_factor = label_dummy.GetSpacing()[::-1]
            label = ndimage.zoom(label, resize_factor, order=0, mode='constant', cval=0.0)
            yield [image, label]

def gen_data(ds):
    while True:
        for d in ds.get_data():
            yield d

class Preprocess(object):
    def __init__(self, path, task):
        self.path = path
        self.task = task
    
        with open(os.path.join(self.path, self.task, 'dataset.json')) as f:
            self.data = json.load(f)
        self.num_labels = len(self.data['labels'])-1
        self.num_channels = len(self.data['modality'])

        '''if self.data['modality']['0'] == 'CT':
                                    self.roi_mean, self.roi_sd = self.CT_calculate_mean_sd(self.path, self.task)
                                    self.center = self.roi_mean
                                    self.width = self.roi_sd * 3 * 2
                                    print('roi_mean : ', roi_mean, ' roi_sd : ', roi_sd, '\nwindowing level is mean and width is 3 sigma * 2')'''
        if self.task == 'Task03_Liver':
            self.center = 100.
            self.width = 180.
            self.mean_sd = [[0.09377181, 0.22647949]]
        elif self.task == 'Task06_Lung':
            self.center = -271.
            self.width = 1716.
            self.mean_sd = [[0.31397295, 0.27987614]]
        elif self.task == 'Task07_Pancreas':
            self.center = 80.
            self.width = 342.
            self.mean_sd = [[0.123380184, 0.23453644]]

        self.datagen = keras_image3d.ImageDataGenerator(
                                        rotation_range=[5.,5.,5.],
                                        zoom_range=[0.9, 1.1],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        depth_shift_range=0.1,
                                        horizontal_flip=False,
                                        fill_mode='constant',cval=0.)

    def CT_calculate_mean_sd(self, path, task):
        train_paths = glob.glob(os.path.join(path, task, 'imagesTr', '*.nii.gz'))
        
        roi_mean = []
        roi_sd = []
        for train_path in train_paths:
            image_header = sitk.ReadImage(train_path)
            image = sitk.GetArrayFromImage(image_header)

            label_header = sitk.ReadImage(train_path.replace('imagesTr','labelsTr'))
            label = sitk.GetArrayFromImage(label_header)
            
            roi_mean.append(np.mean(image[label != 0]))
            roi_sd.append(np.std(image[label != 0]))
        roi_mean = np.mean(np.array(roi_mean))
        roi_sd = np.mean(np.array(roi_sd))
        return roi_mean, roi_sd

    def preprocess_img(self, img):
        if len(self.data['modality']) == 1: 
            img = np.expand_dims(img, -1)
        else:
            img = img.transpose((1,2,3,0)) # channels last

        for self.num_channel in range(self.num_channels):
            if self.data['modality']['0'] == 'CT':
                low = self.center - self.width / 2
                high = self.center + self.width / 2
            else:
                p99 = np.percentile(img[...,self.num_channel], 99.99)
                img[...,self.num_channel] = exposure.rescale_intensity(img[...,self.num_channel], in_range=(0, p99))
                low = np.min(img[...,self.num_channel])
                high = np.max(img[...,self.num_channel])
            img[...,self.num_channel] = (img[...,self.num_channel]-low) / (high-low)
            img[...,self.num_channel][img[...,self.num_channel] < 0.] = 0
            img[...,self.num_channel][img[...,self.num_channel] > 1.] = 1
        return img
    
    def sample_z_norm(self, img):
        if self.data['modality']['0'] == 'CT':
            img -= self.mean_sd[0][0]
            img /= self.mean_sd[0][1]
        else:
            for self.num_channel in range(self.num_channels):
                img[...,self.num_channel] -= np.mean(img[...,self.num_channel])
                img[...,self.num_channel] /= np.std(img[...,self.num_channel])
        return img

    def simple_preprocess_img(self, img):
        if len(self.data['modality']) == 1: 
            img = np.expand_dims(img, -1)
        else:
            img = img.transpose((1,2,3,0)) # channels last
        return img
    
    def simple_sample_z_norm(self, img):
        for self.num_channel in range(self.num_channels):
            img[...,self.num_channel] -= np.mean(img[...,self.num_channel])
            img[...,self.num_channel] /= np.std(img[...,self.num_channel])
        return img

    def simple_preprocess_mask(self, mask):
        mask_dummy = np.zeros(mask.shape+(self.num_labels,))
        for self.num_label in range(self.num_labels): # except background
            mask_dummy[:,:,:,self.num_label][mask == self.num_label+1] = 1.
        return mask_dummy
    
    def whole_merge_mask(self, mask):
        mask = np.sum(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def resize_whole(self, list_img_mask):
        image = list_img_mask[0].copy()
        label = list_img_mask[1].copy()
        
        max_size = 180
        check_size = (image.shape[0]/max_size)*(image.shape[1]/max_size)*(image.shape[2]/max_size)
        if check_size > 1.:
            reduction_factor = ((max_size*max_size*max_size)/(image.shape[0]*image.shape[1]*image.shape[2]))**(1/3)
            resize_factor = (reduction_factor,reduction_factor,reduction_factor)
            image = ndimage.zoom(image, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
            label = ndimage.zoom(label, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        return [image, label]

    def crop_to_patch(self, list_img_mask):
        mask_dummy = list_img_mask[1].copy()
        mask_dummy = np.sum(mask_dummy, axis = -1)
        coordinates = np.argwhere(mask_dummy != 0)
        start_end = [[np.min(coordinates[:,0]),np.max(coordinates[:,0])],
                     [np.min(coordinates[:,1]),np.max(coordinates[:,1])],
                     [np.min(coordinates[:,2]),np.max(coordinates[:,2])]]
        
        x_start = start_end[0][0]
        x_end = start_end[0][1]
        y_start = start_end[1][0]
        y_end = start_end[1][1]
        z_start = start_end[2][0]
        z_end = start_end[2][1]
        
        # define margin for minimum size
        min_size = 24
        if x_end-x_start < min_size:
            x_margin = math.ceil((min_size-(x_end-x_start))/2)
            if x_start-x_margin < 0:
                x_start = 0
            else:
                x_start -= x_margin
            if x_end+x_margin > mask_dummy.shape[0]:
                x_end = mask_dummy.shape[0]
            else:
                x_end += x_margin
        if y_end-y_start < min_size:
            y_margin = math.ceil((min_size-(y_end-y_start))/2)
            if y_start-y_margin < 0:
                y_start = 0
            else:
                y_start -= y_margin
            if y_end+y_margin > mask_dummy.shape[1]:
                y_end = mask_dummy.shape[1]
            else:
                y_end += y_margin
        if z_end-z_start < min_size:
            z_margin = math.ceil((min_size-(z_end-z_start))/2)
            if z_start-z_margin < 0:
                z_start = 0
            else:
                z_start -= z_margin
            if z_end+z_margin > mask_dummy.shape[2]:
                z_end = mask_dummy.shape[2]
            else:
                z_end += z_margin
        
        # define margin 20% of object size
        x_margin = math.ceil((x_end-x_start)*0.2)
        y_margin = math.ceil((y_end-y_start)*0.2)
        z_margin = math.ceil((z_end-z_start)*0.2)
        if x_start-x_margin < 0:
            x_start = 0
        else:
            x_start -= x_margin
        if x_end+x_margin > mask_dummy.shape[0]:
            x_end = mask_dummy.shape[0]
        else:
            x_end += x_margin

        if y_start-y_margin < 0:
            y_start = 0
        else:
            y_start -= y_margin
        if y_end+y_margin > mask_dummy.shape[1]:
            y_end = mask_dummy.shape[1]
        else:
            y_end += y_margin
        
        if z_start-z_margin < 0:
            z_start = 0
        else:
            z_start -= z_margin
        if z_end+z_margin > mask_dummy.shape[2]:
            z_end = mask_dummy.shape[2]
        else:
            z_end += z_margin

        list_img_mask[0] = list_img_mask[0][x_start:x_end,y_start:y_end,z_start:z_end,:]
        list_img_mask[1] = list_img_mask[1][x_start:x_end,y_start:y_end,z_start:z_end,:]
        return list_img_mask

    def resize_patch(self, list_img_mask):
        crop_image = list_img_mask[0].copy()
        crop_label = list_img_mask[1].copy()
        
        resize_factor = ()
        for num_axis in range(3):
            if crop_image.shape[num_axis] < 24:
                resize_factor += (24./crop_image.shape[num_axis],)
            else:
                resize_factor += (1.,)
        if resize_factor != (1., 1., 1.):
            crop_image = ndimage.zoom(crop_image, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
            crop_label = ndimage.zoom(crop_label, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
        
        max_size = 180
        check_size = (crop_image.shape[0]/max_size)*(crop_image.shape[1]/max_size)*(crop_image.shape[2]/max_size)
        if check_size > 1.:
            reduction_factor = ((max_size*max_size*max_size)/(crop_image.shape[0]*crop_image.shape[1]*crop_image.shape[2]))**(1/3)
            resize_factor = (reduction_factor,reduction_factor,reduction_factor)
            crop_image = ndimage.zoom(crop_image, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
            crop_label = ndimage.zoom(crop_label, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
        crop_image = np.expand_dims(crop_image, 0)
        crop_label = np.expand_dims(crop_label, 0)
        return [crop_image, crop_label]

    def data_aug(self, list_img_mask):
        seed = np.random.randint(0, 100, 1)[0]
        for img in self.datagen.flow(list_img_mask[0], batch_size=1, seed = seed):
            break
        for mask in self.datagen.flow(list_img_mask[1], batch_size=1, seed = seed):
            break
        return [img, mask]