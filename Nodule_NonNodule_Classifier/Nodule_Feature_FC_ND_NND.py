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


class Nodule_FC():

    def __init__(self):
        Feature_input = Input(shape=(239,), name='batch_train_feature')

        x = Dense(128, activation='relu', init='he_normal')(Feature_input)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        nodules = Dense(2, activation='softmax', name="nodule")(x)


        self.model = Model(input=[Feature_input], output=[nodules])
        #self.model = Model(input=[img_input, size_input], output=[batch_feature_label])
        self.model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                           loss='binary_crossentropy', metrics=['accuracy'])




