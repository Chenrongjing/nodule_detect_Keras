#import MODEL_3D.Nodule_3DCNN as CNN
import train_Nodule_gen_dual_2D_XCEPTION_ND_NND as CNN
import csv
import numpy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
import os
import cv2

DATA_FOLDER = '/home/kwanghee/LIDC/dsb2017/KAGGLE_NODULE_CROP_ALL_NPY_MG10/'
CSV_KAGGLE_PATH = '/home/kwanghee/LIDC/dsb2017/stage1_labels.csv'
#CSV_PATH = '/root/workspace/dsb2017_tutorial/3d_conv/lidc/csv/annotations_nodule_label_aug.csv'
CSV_LIDC_TRAINING_PATH = 'annotations_luna_training.csv'
CSV_LIDC_TEST_PATH = 'annotations_luna_test.csv'
#NUMPY_ROOT_PATH = '/root/workspace/dsb2017_tutorial/3d_conv/lidc/classification_crop_all/'
NUMPY_ROOT_PATH = '/home/kwanghee/LIDC/dsb2017/classification_rect2/'
ORI_PATH = "rescale/"
FLIP_HORI_PATH = "rescale_flip/hori/"
FLIP_VERT_PATH = "rescale_flip/vert/"

R_ORI_PATH = "rescale_rotate/ori/"
R_FLIP_HORI_PATH = "rescale_rotate/flip_hori/"
R_FLIP_VERT_PATH = "rescale_rotate/flip_vert/"

AUG_PATH_LIST = [ORI_PATH,FLIP_HORI_PATH,FLIP_VERT_PATH,R_ORI_PATH,R_FLIP_HORI_PATH,R_FLIP_VERT_PATH]
#np.set_printoptions(threshold=np.nan)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.test_losses = []
        self.val_losses = []
        self.val_acc = []
        # self.f_losses = []
        # self.f_val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.test_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        # self.f_losses.append(logs.get('feature_loss'))
        # self.f_val_losses.append(logs.get('val_feature_loss'))
        plt.plot(self.test_losses)
        plt.plot(self.val_losses)
        plt.plot(self.val_acc)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("./Nodule_2DCNN_ND_NND.png")
        plt.close()

        # plt.plot(self.f_losses)
        # plt.plot(self.f_val_losses)
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig("./Nodule_2DCNN_DUAL_xception_multi_label_feature.png")
        # plt.close()

def Kaggle_Data_Load(DATA_FOLDER, CSV_KAGGLE_PATH, Training_Sample , Test_Sample):
    patients = []

    with open(CSV_KAGGLE_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patients.append([row['id'], row['cancer']])

    # patients = os.listdir(INPUT_FOLDER)
    #patients.sort()
    patients.sort(key=lambda x: str(x[0]), reverse=False)
    #print patients

    NonNodule_filelist = []
    #NonNodule_label = []

    for i in xrange(len(patients)):
        if patients[i][1] == str(0):
            nodules = os.listdir(DATA_FOLDER+"V"+str(i)+"/")
            for j in xrange(len(nodules)):
                #print nodules[j]+ "V" + str(i) + "_" + patients[i][0] + "_ND" + str(j) + ".npy"
                NonNodule_filelist.append([DATA_FOLDER+"V"+str(i)+"/"+nodules[j],0])
                #NonNodule_label.append(0)
    #print numpy.shape(NonNodule_filelist)
    #print (NonNodule_filelist[0])
    NonNodule_filelist = shuffle(NonNodule_filelist)

    NonNodule_traing_filelist = NonNodule_filelist[0:Training_Sample]
    NonNodule_test_filelist = NonNodule_filelist[Training_Sample+1:min(Training_Sample+Test_Sample+1, len(NonNodule_filelist)-1)]

    return NonNodule_traing_filelist, NonNodule_test_filelist

def LIDC_Data_Load( NUMPY_ROOT_PATH,CSV_LIDC_TRAINING_PATH, CSV_LIDC_TEST_PATH ,ORI_PATH, FLIP_HORI_PATH, FLIP_VERT_PATH, AUG_PATH_LISTl ):

    Nodule_training_filelist = []
    Nodule_test_filelist = []

    with open(CSV_LIDC_TRAINING_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = NUMPY_ROOT_PATH+AUG_PATH_LIST[int(row['aug'])-1]+row['pt_id']+"&"+row['n_id']+".npy"
            data = [filename, 1]
            Nodule_training_filelist.append(data)
    #print Nodule_training_filelist[0]

    #print numpy.shape(Nodule_training_filelist)
    Nodule_training_filelist = shuffle(Nodule_training_filelist)

    with open(CSV_LIDC_TEST_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = NUMPY_ROOT_PATH+AUG_PATH_LIST[int(row['aug'])-1]+row['pt_id']+"&"+row['n_id']+".npy"
            data = [filename, 1]
            Nodule_test_filelist.append(data)
    #print Nodule_test_filelist[0]

    #print numpy.shape(Nodule_test_filelist)
    Nodule_test_filelist = shuffle(Nodule_test_filelist)

    return Nodule_training_filelist, Nodule_test_filelist


def make_diagonal_image(vol, FIX_LEN = 70):
    center = int(FIX_LEN/2)
    rotate = 45
    M = cv2.getRotationMatrix2D((center, center), rotate, 1)
    diagonal_vol = []
    for i in range(0,FIX_LEN):
        diagonal_vol.append(cv2.warpAffine(vol[i], M, (FIX_LEN, FIX_LEN), borderValue=-2000))

    diagonal_vol = numpy.array(diagonal_vol, dtype=numpy.float32)
    return numpy.transpose(diagonal_vol,(1, 2, 0))[35], numpy.transpose(diagonal_vol,(2,1,0))[35]

def generate_batch(data_list, batch_size ):

    data_list_len = len(data_list)
    #print data_list_len
    count = 0
    while True:
        batch_train_image = []
        batch_train_label = []
        #for count in range(0, loop_num):
        if (count+1)*batch_size <= data_list_len:
            max_len = (count+1)*batch_size
        else:
            max_len = data_list_len
        batch_data = data_list[count*batch_size:max_len]
        for data in batch_data:
            if data[1] == 1:
                batch_train_label.append([1, 0])
            else:
                batch_train_label.append([0, 1])

            file_path = data[0]
            train_image = numpy.load(file_path)
            train_image = numpy.array(train_image,dtype=numpy.float32)

            min = -1000
            max = train_image.max()
            train_image = (train_image - min) / (max - min)

            train_image_2d = []
            train_image = (train_image*2) - 1
            train_image[train_image < -1] = -1
            train_image[train_image > 1] = 1

            nodule_Img1 = train_image[int(round(numpy.shape(train_image)[0] / 2)), :, :]
            nodule_Img2 = train_image[:, int(round(numpy.shape(train_image)[0] / 2)), :]
            nodule_Img3 = train_image[:, :, int(round(numpy.shape(train_image)[0] / 2))]

            nodule_Img4, nodule_Img5 = make_diagonal_image(train_image)
            nodule_Img6, nodule_Img7 = make_diagonal_image(numpy.transpose(train_image, (1, 2, 0)))
            nodule_Img8, nodule_Img9 = make_diagonal_image(numpy.transpose(train_image, (2, 1, 0)))

            train_image_2d.append(nodule_Img1)
            train_image_2d.append(nodule_Img2)
            train_image_2d.append(nodule_Img3)

            train_image_2d.append(nodule_Img4)
            train_image_2d.append(nodule_Img5)
            train_image_2d.append(nodule_Img6)

            train_image_2d.append(nodule_Img7)
            train_image_2d.append(nodule_Img8)
            train_image_2d.append(nodule_Img9)

            train_image_2d = numpy.array(train_image_2d)
            train_image_2d = numpy.transpose(train_image_2d, (1,2,0))
            batch_train_image.append(train_image_2d)

        batch_train_image = numpy.array(batch_train_image)

        count = count + 1
        if count*batch_size >= data_list_len:
            count = 0
            data_list = shuffle(data_list)

        yield batch_train_image, batch_train_label

if __name__ == '__main__':


    NonNodule_traing_filelist, NonNodule_test_filelist = Kaggle_Data_Load(DATA_FOLDER, CSV_KAGGLE_PATH, Training_Sample = 30000, Test_Sample = 10000)
    Nodule_training_filelist, Nodule_test_filelist = LIDC_Data_Load(NUMPY_ROOT_PATH, CSV_LIDC_TRAINING_PATH,CSV_LIDC_TEST_PATH,
                         ORI_PATH, FLIP_HORI_PATH, FLIP_VERT_PATH, AUG_PATH_LIST)

    print numpy.shape(NonNodule_traing_filelist)
    print numpy.shape(NonNodule_test_filelist)
    print numpy.shape(Nodule_training_filelist)
    print numpy.shape(Nodule_test_filelist)

    Training_filelist = Nodule_training_filelist + NonNodule_traing_filelist
    Test_filelist = Nodule_test_filelist + NonNodule_test_filelist

    Training_filelist = shuffle(Training_filelist)
    Test_filelist = shuffle(Test_filelist)


    # print Training_filelist[0:10]
    # print Test_filelist[0:10]

    nodule_model = CNN.Nodule_2DCNN()
    nodule_model.model.summary()
    history = LossHistory()
    #data_lists = data_lists[:100]
    #if os.path.isfile('./nodule_pad_best_bn.hdf5'):
    #    print ("load ./nodule_pad_best_bn.hdf5")
    #    nodule_model.model = load_model('./nodule_rescale_best_bn.hdf5')
    callbacks = [
        #EarlyStopping(monitor='val_loss', patience=10000, verbose=1),
        history,
        ModelCheckpoint('Nodule_2DCNN_ND_NND.hdf5', monitor='val_loss', save_best_only=True, verbose=1),
        #ModelCheckpoint('Nodule_2DCNN_DUAL_xception_multi_label_feature.hdf5', monitor='val_feature_loss', save_best_only=True, verbose=1)
    ]


    batch_size = 15

    epoch = 99999999999999
    samples_per_epoch = len(Training_filelist)
    nb_val_samples = len(Test_filelist)
    print (samples_per_epoch, nb_val_samples)
    history = nodule_model.model.fit_generator(generator=generate_batch(Training_filelist, batch_size),
                          nb_epoch=epoch,
                          samples_per_epoch=samples_per_epoch,
                          validation_data=generate_batch(Test_filelist, batch_size),
                          nb_val_samples=nb_val_samples,
                          verbose=1,
                          callbacks=callbacks)
