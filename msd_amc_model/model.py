
# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Lambda, LeakyReLU, Multiply, Reshape
from tensorflow.python.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose, GlobalAveragePooling3D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.layers import BatchNormalization, GaussianNoise
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras import backend as K
import memory_saving_gradients
from msd_amc_preprocess import keras_image3d
K.__dict__["gradients"] = memory_saving_gradients.gradients_speed

smooth = 1.
def average_dice_coef(y_true, y_pred):
    loss = 0
    label_length = y_pred.get_shape().as_list()[-1]
    for num_label in range(label_length):
        y_true_f = K.flatten(y_true[...,num_label])
        y_pred_f = K.flatten(y_pred[...,num_label])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss / label_length # 1>= loss >0

def average_dice_coef_loss(y_true, y_pred):
    return -average_dice_coef(y_true, y_pred)

def load_model(input_shape, num_labels, base_filter=32, se_block=True, se_ratio=16, noise=0.1):
    def conv3d(layer_input, filters, axis=-1, se_block=True, se_ratio=16, down_sizing=True):
        if down_sizing == True:
            layer_input = MaxPooling3D(pool_size=(2, 2, 2))(layer_input)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
        d = BatchNormalization(axis=axis)(d)
        d = LeakyReLU(alpha=0.3)(d)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(d)
        d = BatchNormalization(axis=axis)(d)
        if se_block == True:
            se = GlobalAveragePooling3D()(d)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            d = Multiply()([d, se])
        d = LeakyReLU(alpha=0.3)(d)
        return d

    def deconv3d(layer_input, skip_input, filters, axis=-1, se_block=True, se_ratio=16):
        u = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(layer_input)
        u = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u)
        u = BatchNormalization(axis=axis)(u)
        u = LeakyReLU(alpha=0.3)(u)
        u = CropToConcat3D()([u, skip_input])
        u = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u)
        u = BatchNormalization(axis = axis)(u)
        u = LeakyReLU(alpha=0.3)(u)
        u = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u)
        u = BatchNormalization(axis = axis)(u)
        if se_block == True:
            se = GlobalAveragePooling3D()(u)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            u = Multiply()([u, se])
        u = LeakyReLU(alpha=0.3)(u)
        return u

    def CropToConcat3D():
        def crop_to_concat_3D(concat_layers, axis=-1):
            bigger_input,smaller_input = concat_layers
            bigger_shape, smaller_shape = tf.shape(bigger_input), \
                                          tf.shape(smaller_input)
            sh,sw,sd = smaller_shape[1],smaller_shape[2],smaller_shape[3]
            bh,bw,bd = bigger_shape[1],bigger_shape[2],bigger_shape[3]
            dh,dw,dd = bh-sh, bw-sw, bd-sd
            cropped_to_smaller_input = bigger_input[:,:-dh,
                                                      :-dw,
                                                      :-dd,:]
            return K.concatenate([smaller_input,cropped_to_smaller_input], axis=axis)
        return Lambda(crop_to_concat_3D)

    input_img = Input(shape=input_shape)
    d0 = GaussianNoise(noise)(input_img)
    d1 = conv3d(d0, base_filter, se_block=False, down_sizing=False)
    d2 = conv3d(d1, base_filter*2, se_block=se_block)
    d3 = conv3d(d2, base_filter*4, se_block=se_block)
    d4 = conv3d(d3, base_filter*8, se_block=se_block)
    d5 = conv3d(d4, base_filter*16, se_block=se_block)

    u4 = deconv3d(d5, d4, base_filter*8)
    u3 = deconv3d(u4, d3, base_filter*4)
    u2 = deconv3d(u3, d2, base_filter*2)
    u1 = deconv3d(u2, d1, base_filter, se_block=False)
    output_img = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', activation='sigmoid')(u1)
    model = Model(inputs=input_img, outputs=output_img)
    return model

def data_aug(task):
    if task == 'Task04_Hippocampus':
        rotation_range = [3.,3.,3.]
    elif task == 'Task05_Prostate':
        rotation_range = [0.,0.,10.]
    else:
        rotation_range = [5.,5.,10.]
    if task == 'Task01_BrainTumour' or task == 'Task05_Prostate' or task == 'Task06_Lung':
        horizontal_flip = True
    else:
        horizontal_flip = False
    datagen = keras_image3d.ImageDataGenerator(
                                        rotation_range=rotation_range, #####################
                                        zoom_range=[0.9, 1.1],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        depth_shift_range=0.1,
                                        horizontal_flip=horizontal_flip,
                                        fill_mode='constant',cval=0.)
    return datagen