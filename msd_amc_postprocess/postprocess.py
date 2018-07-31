import numpy as np
import math

def crop_to_patch(result, image):
    mask_dummy = result.copy()
    coordinates = np.argwhere(mask_dummy != 0)
    start_end = [[np.min(coordinates[:,0]),np.max(coordinates[:,0])],
                 [np.min(coordinates[:,1]),np.max(coordinates[:,1])],
                 [np.min(coordinates[:,2]),np.max(coordinates[:,2])]]
    
    x_start = int(start_end[0][0] * (image.shape[0]/mask_dummy.shape[0]))
    x_end = math.ceil(start_end[0][1] * (image.shape[0]/mask_dummy.shape[0]))
    y_start = int(start_end[1][0] * (image.shape[1]/mask_dummy.shape[1]))
    y_end = math.ceil(start_end[1][1] * (image.shape[1]/mask_dummy.shape[1]))
    z_start = int(start_end[2][0] * (image.shape[2]/mask_dummy.shape[2]))
    z_end = math.ceil(start_end[2][1] * (image.shape[2]/mask_dummy.shape[2]))
    
    # define margin for minimum size
    min_size = 24
    if x_end-x_start < min_size:
        x_margin = math.ceil((min_size-(x_end-x_start))/2)
        if x_start-x_margin < 0:
            x_start = 0
        else:
            x_start -= x_margin
        if x_end+x_margin > image.shape[0]:
            x_end = image.shape[0]
        else:
            x_end += x_margin
    if y_end-y_start < min_size:
        y_margin = math.ceil((min_size-(y_end-y_start))/2)
        if y_start-y_margin < 0:
            y_start = 0
        else:
            y_start -= y_margin
        if y_end+y_margin > image.shape[1]:
            y_end = image.shape[1]
        else:
            y_end += y_margin
    if z_end-z_start < min_size:
        z_margin = math.ceil((min_size-(z_end-z_start))/2)
        if z_start-z_margin < 0:
            z_start = 0
        else:
            z_start -= z_margin
        if z_end+z_margin > image.shape[2]:
            z_end = image.shape[2]
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
    if x_end+x_margin > image.shape[0]:
        x_end = image.shape[0]
    else:
        x_end += x_margin

    if y_start-y_margin < 0:
        y_start = 0
    else:
        y_start -= y_margin
    if y_end+y_margin > image.shape[1]:
        y_end = image.shape[1]
    else:
        y_end += y_margin
    
    if z_start-z_margin < 0:
        z_start = 0
    else:
        z_start -= z_margin
    if z_end+z_margin > image.shape[2]:
        z_end = image.shape[2]
    else:
        z_end += z_margin

    image = image[x_start:x_end,y_start:y_end,z_start:z_end,:]
    return image, x_start, x_end, y_start, y_end, z_start, z_end

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