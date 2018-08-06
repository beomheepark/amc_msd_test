
# coding: utf-8

import os
import glob
import numpy as np
from scipy import ndimage
import json
import SimpleITK as sitk
from skimage import exposure
from skimage import measure
import math
import random
from msd_amc_preprocess2 import keras_image3d
from tensorpack.dataflow.base import RNGDataFlow
from skimage.exposure import adjust_gamma

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
    def __init__(self, path, task, cal_CT_params=False, phase='valid'):
        self.path = path
        self.task = task
        self.phase = phase
        with open(os.path.join(self.path, self.task, 'dataset.json')) as f:
            self.data = json.load(f)
        self.num_labels = len(self.data['labels'])-1
        self.num_channels = len(self.data['modality'])

        if cal_CT_params == True:
            if self.data['modality']['0'] == 'CT':
                self.roi_mean, self.roi_sd = self.cal_CT_win_params(self.path, self.task)
                self.center = self.roi_mean
                self.width = self.roi_sd * 3 * 2
                print('roi_mean : ', self.roi_mean, ' roi_sd : ', self.roi_sd, '\nwindowing level is mean and width is 3 sigma * 2')
                self.mean_sd = self.cal_CT_z_params(self.path, self.task)
                print('image mean & sd : ', self.mean_sd)


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
                                        rotation_range=[10.,10.,10.],
                                        zoom_range=[0.9, 1.1],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        depth_shift_range=0.1,
                                        shear_range=0.2,
                                        fill_mode='constant',cval=0.)

    def cal_CT_win_params(self, path, task):
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

    def cal_CT_z_params(self, path, task):
        train_paths = glob.glob(os.path.join(path, task, 'imagesTr', '*.nii.gz'))
        mean = []
        sd = []
        for train_path in train_paths:
            image_header = sitk.ReadImage(train_path)
            image = sitk.GetArrayFromImage(image_header)
            image = self.preprocess_img(image)
            mean.append(np.mean(image))
            sd.append(np.std(image))
        return [[np.mean(np.array(mean)), np.mean(np.array(sd))]]

    def random_gamma(self, img):
        gamma = np.random.random()+0.5
        img = adjust_gamma(img, gamma=gamma)
        return img

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

    def z_normalization(self, img):
        if self.data['modality']['0'] == 'CT':
            img -= self.mean_sd[0][0]
            img /= self.mean_sd[0][1]
        else:
            for self.num_channel in range(self.num_channels):
                img[...,self.num_channel] -= np.mean(img[...,self.num_channel])
                img[...,self.num_channel] /= np.std(img[...,self.num_channel])
        return img

    def merge_mask(self, mask):
        mask = np.sum(mask, axis=-1)
        mask[mask != 0] = 1.
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def preprocess_mask(self, mask):
        mask_dummy = np.zeros(mask.shape+(self.num_labels,))
        for self.num_label in range(self.num_labels): # except background
            mask_dummy[:,:,:,self.num_label][mask == self.num_label+1] = 1.
        return mask_dummy

    def resize_to_limit(self, list_img_mask):
        image = list_img_mask[0].copy()
        label = list_img_mask[1].copy()
        
        resize_factor = ()
        for num_axis in range(3):
            if image.shape[num_axis] < 24:
                resize_factor += (24./image.shape[num_axis],)
            else:
                resize_factor += (1.,)
        if resize_factor != (1., 1., 1.):
            image = ndimage.zoom(image, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
            label = ndimage.zoom(label, resize_factor+(1.,), order=0, mode='constant', cval=0.0)
        
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
        if self.task == 'Task01_BrainTumour':
            mask_dummy = ndimage.binary_dilation(mask_dummy, structure=np.ones((5,5,5))).astype(mask_dummy.dtype)
            mask_dummy = ndimage.binary_erosion(mask_dummy, structure=np.ones((5,5,5))).astype(mask_dummy.dtype)
            mask_dummy = measure.label(mask_dummy, background = 0)
            clusters = np.unique(mask_dummy)
            clusters.sort()
            clusters = clusters[1:]
            clusters = np.random.choice(clusters)
            mask_dummy[mask_dummy != clusters] = 0
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
        
        if self.phase == 'train':
            x_margin = math.ceil((x_end-x_start)*(np.random.randint(1, 4)*0.1))
            y_margin = math.ceil((y_end-y_start)*(np.random.randint(1, 4)*0.1))
            z_margin = math.ceil((z_end-z_start)*(np.random.randint(1, 4)*0.1))
        elif self.phase == 'valid':
            x_margin = math.ceil((x_end-x_start)*0.2)
            y_margin = math.ceil((y_end-y_start)*0.2)
            z_margin = math.ceil((z_end-z_start)*0.2)
        else:
            raise PhaseInputError('phase must be train or valid, but got {}.'.format(self.phase))

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
        
        if self.phase == 'train':
            if np.random.random() <= 0.8:
                list_img_mask[0] = list_img_mask[0][x_start:x_end,y_start:y_end,z_start:z_end,:]
                list_img_mask[1] = list_img_mask[1][x_start:x_end,y_start:y_end,z_start:z_end,:]
            else:
                rand_x_start = mask_dummy.shape[0]-(x_end-x_start)+1
                rand_y_start = mask_dummy.shape[1]-(y_end-y_start)+1
                rand_z_start = mask_dummy.shape[2]-(z_end-z_start)+1
                list_img_mask[0] = list_img_mask[0][rand_x_start:rand_x_start+(x_end-x_start),rand_y_start:rand_y_start+(y_end-y_start),rand_z_start:rand_z_start+(z_end-z_start),:]
                list_img_mask[1] = list_img_mask[1][rand_x_start:rand_x_start+(x_end-x_start),rand_y_start:rand_y_start+(y_end-y_start),rand_z_start:rand_z_start+(z_end-z_start),:]
        elif self.phase == 'valid':
            list_img_mask[0] = list_img_mask[0][x_start:x_end,y_start:y_end,z_start:z_end,:]
            list_img_mask[1] = list_img_mask[1][x_start:x_end,y_start:y_end,z_start:z_end,:]
        else:
            raise PhaseInputError('phase must be train or valid, but got {}.'.format(self.phase))
        return list_img_mask

    def data_aug(self, list_img_mask):
        seed = np.random.randint(0, 100, 1)[0]
        for img in self.datagen.flow(list_img_mask[0], batch_size=1, seed = seed):
            break
        for mask in self.datagen.flow(list_img_mask[1], batch_size=1, seed = seed):
            break
        return [img, mask]

def label_encoding(result, num_labels):
    result_dummy = np.zeros(result.shape[:-1])
    for num_label in range(num_labels):
        result_dummy[result[...,num_label] != 0] = num_label+1
    return result_dummy

def reverse_label_encoding(result, num_labels):
    result_dummy = np.zeros(result.shape[:-1])
    for num_label in reversed(range(num_labels)):
        result_dummy[result[...,num_label] != 0] = num_label+1
    return result_dummy
