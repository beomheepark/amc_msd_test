
# coding: utf-8

import os
import glob
import json
import random
import time
import SimpleITK as sitk
import numpy as np
from msd_amc_preprocess import preprocess
from msd_amc_postprocess import postprocess
from msd_amc_model.model import average_dice_coef, average_dice_coef_loss, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow import PrefetchData
from scipy import ndimage
import argparse
from skimage import measure
import math
import nibabel as nib

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = "/data/public/rw/medical/decathlon2018_rev0704"
tasks = ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate','Task06_Lung','Task07_Pancreas']

save_path = "/data/private/beomheep/00.MSD/results_ensemble_sum_average"

for task in tasks: # tasks
    test_cases = glob.glob(os.path.join(path, task, 'imagesTs', '*'))
    print('# of test data : ', len(test_cases))

    config_task = preprocess.Preprocess(path, task)
    input_shape = (None, None, None, config_task.num_channels)

    num_channels = config_task.num_channels
    model_whole = load_model(input_shape=input_shape,
                        num_labels=1,
                        base_filter=32,
                        depth_size=3,
                        se_res_block=True,
                        se_ratio=16,
                        last_relu=True)

    model_patch = load_model(input_shape=input_shape,
                        num_labels=config_task.num_labels,
                        base_filter=32,
                        depth_size=3,
                        se_res_block=True,
                        se_ratio=16,
                        last_relu=True)
    
    try:
        os.stat(os.path.join(save_path, task))
    except:
        os.mkdir(os.path.join(save_path, task))
        
    for test_case in test_cases:
        image_dummy = sitk.ReadImage(test_case)
        image_origin = sitk.GetArrayFromImage(image_dummy).astype('float32')
        resize_factor = image_dummy.GetSpacing()[::-1]
        image_origin = ndimage.zoom(image_origin, resize_factor, order=0, mode='constant', cval=0.0)

        image_origin = config_task.preprocess_img(image_origin)
        image_origin = config_task.sample_z_norm(image_origin)
        ##############################
        # whole image
        ##############################        
        image = image_origin.copy()
        image, _ = config_task.resize_whole([image, image])
        
        whole_result = np.zeros(image.shape[:4]+(1,))
        for split_index in range(0,4,1):
            model_whole.load_weights('/root/beom/00_MSD_predict/180726_{}_{}_{}.h5'.format(task,'whole',split_index)) #######################
            whole_result += model_whole.predict(image)
            #break ###################
        #whole_result /= 4 # average ######################
        if len(whole_result[whole_result >= 0.5]) == 0:
            whole_result[whole_result == np.max(whole_result)] = 1.
        whole_result = np.squeeze(whole_result)
        whole_result[whole_result < 0.5] = 0
        whole_result[whole_result >= 0.5] = 1
        
        ############################# connectivity (one cluster)
        whole_result = measure.label(whole_result, background = 0)
        clusters = np.unique(whole_result)
        clusters.sort()
        max_cluster = 0
        max_size = 0
        for cluster in clusters[1:]: # except '0' : background
            if len(whole_result[whole_result == cluster]) > max_size:
                max_size = len(whole_result[whole_result == cluster])
                max_cluster = cluster
        whole_result[whole_result != max_cluster] = 0
        
        ##############################
        # patch image
        ##############################
        image_patch, x_start, x_end, y_start, y_end, z_start, z_end = postprocess.crop_to_patch(whole_result, image_origin)
        image_patch, _ = config_task.resize_patch([image_patch, image_patch])
        
        patch_result = np.zeros(image_patch.shape[:4]+(config_task.num_labels,))
        for split_index in range(0,4,1):
            model_patch.load_weights('/root/beom/00_MSD_predict/180726_{}_{}_{}.h5'.format(task,'patch',split_index))
            patch_result += model_patch.predict(image_patch)
            #break ###################
        patch_result /= 4 # average ######################
        patch_result[patch_result < 0.5] = 0
        patch_result[patch_result >= 0.5] = 1
        patch_result = patch_result[0]

        patch_result = postprocess.label_encoding(patch_result, config_task.num_labels)
        reverse_resize_factor = ((x_end-x_start)/patch_result.shape[0], (y_end-y_start)/patch_result.shape[1], (z_end-z_start)/patch_result.shape[2])
        patch_result = ndimage.zoom(patch_result, reverse_resize_factor, order=0, mode='constant', cval=0.0)
        
        save_mask = np.zeros(image_origin.shape[:3])
        save_mask[x_start:x_end, y_start:y_end, z_start:z_end] = patch_result
        save_mask = save_mask.astype(np.uint8)
        reverse_resize_factor = (image_dummy.GetDepth()/save_mask.shape[0], image_dummy.GetHeight()/save_mask.shape[1], image_dummy.GetWidth()/save_mask.shape[2])
        save_mask = ndimage.zoom(save_mask, reverse_resize_factor, order=0, mode='constant', cval=0.0)
        if save_mask.shape != (image_dummy.GetDepth(), image_dummy.GetHeight(), image_dummy.GetWidth()):
            print(test_case)
        save_mask = save_mask.transpose(range(len(save_mask.shape))[::-1])
        ##############################
        # save
        ##############################
        nib_headers = nib.load(test_case)
        affine = np.eye(4)
        affine[:3,:] = np.array([nib_headers.header['srow_x'], nib_headers.header['srow_y'], nib_headers.header['srow_z']])
        nib_save = nib.Nifti1Image(save_mask, affine, header=nib_headers.header)
        nib.save(nib_save, os.path.join(save_path, task, test_case.split('/')[-1]))
    print(task, 'done')

