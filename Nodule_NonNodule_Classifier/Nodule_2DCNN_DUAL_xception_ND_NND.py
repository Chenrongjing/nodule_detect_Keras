from keras.utils.data_utils import get_file
import tensorflow as tf
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras import layers
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import merge
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.preprocessing import image
from keras.layers import Merge, Input
from keras.layers.advanced_activations import LeakyReLU

CONV1_FILTER = [3,3]
CONV2_FILTER = [3,3]
CONV3_FILTER = [3,3]
CONV4_FILTER = [3,3]
CLASS_NUM = 2
ROW = 70
COL = 70
CHANNEL = 9

class Nodule_2DCNN():

    def __init__(self):
        img_input = Input(shape=(ROW, COL, CHANNEL), name='batch_train_image')
        #size_input= Input(shape=(3,), name='size_list')

        x = Convolution2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1', init='he_normal')(img_input)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Convolution2D(64, 3, 3, bias=False, name='block1_conv2', init='he_normal')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Convolution2D(128, 1, 1, subsample=(2, 2),
                                 border_mode='same', bias=False, init='he_normal')(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1', init='he_normal')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2', init='he_normal')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
        x = merge([x, residual], mode='sum')

        residual = Convolution2D(256, 1, 1, subsample=(2, 2),
                                 border_mode='same', bias=False, init='he_normal')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1', init='he_normal')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2', init='he_normal')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
        #x = layers.add([x, residual])
        x = merge([x, residual], mode='sum')

        residual = Convolution2D(1024, 1, 1, subsample=(2, 2),
                                 border_mode='same', bias=False, init='he_normal')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block4_sepconv1', init='he_normal')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block4_sepconv2', init='he_normal')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
        x = merge([x, residual], mode='sum')

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1', init='he_normal')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2', init='he_normal')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3', init='he_normal')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

            x = merge([x, residual], mode='sum')

        residual = Convolution2D(1536, 1, 1, subsample=(2, 2),
                                 border_mode='same', bias=False, init='he_normal')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block13_sepconv1_act')(x)
        x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv1', init='he_normal')(x)
        x = BatchNormalization(name='block13_sepconv1_bn')(x)
        x = Activation('relu', name='block13_sepconv2_act')(x)
        x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block13_sepconv2', init='he_normal')(x)
        x = BatchNormalization(name='block13_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
        x = merge([x, residual], mode='sum')

        x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv1', init='he_normal')(x)
        x = BatchNormalization(name='block14_sepconv1_bn')(x)
        x = Activation('relu', name='block14_sepconv1_act')(x)

        x = SeparableConv2D(4096, 3, 3, border_mode='same', bias=False, name='block14_sepconv2', init='he_normal')(x)
        x = BatchNormalization(name='block14_sepconv2_bn')(x)
        x = Activation('relu', name='block14_sepconv2_act')(x)

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(2048, activation='relu', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        nodules = Dense(2, activation='softmax', name="nodule")(x)
        #malignancy_feature = merge([x, size_input], mode='concat')
        # malignancy_feature = merge([batch_feature_label, size_input], mode='concat')
        # batch_cate_label = Dense(1, activation='linear', name="malignancy")(malignancy_feature)


        self.model = Model(input=[img_input], output=[nodules])
        #self.model = Model(input=[img_input, size_input], output=[batch_feature_label])
        self.model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                           loss='binary_crossentropy', metrics=['accuracy'])




