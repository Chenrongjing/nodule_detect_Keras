from skimage.measure import label,regionprops
import numpy as np # linear algebra

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from sklearn.decomposition import PCA as sklearnPCA
import os
import cv2
import matplotlib.pyplot as plt

def Global_Thresholding_2D(img):
    mask_img = np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype=int)
    temp_img = np.copy(img)

    box_array = []
    for i in xrange(0, np.shape(img)[0]):
        for j in xrange(0, np.shape(img)[1]):
                if temp_img[i,j]>-1000 and temp_img[i,j]<100:
                    box_array.append(temp_img[i,j])

    box_array_np = np.asarray(box_array)

    if (np.shape(box_array_np)[0] >0):
        mask_img[img > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = 1
        mask_img[img > 100] = 0


    return mask_img

def Global_Thresholding_3D(vol):
    mask_vol = np.zeros((np.shape(vol)[0], np.shape(vol)[1], np.shape(vol)[2]), dtype = int)
    temp_vol = np.copy(vol)
    #temp_vol[vol>-500] = -2000
    box_array = []
    for i in xrange(0, np.shape(vol)[0],4):
        for j in xrange(0, np.shape(vol)[1],4):
            for k in xrange(0, np.shape(vol)[2],4):
                if temp_vol[i,j,k]>-1000 and temp_vol[i,j,k]<100:
                    box_array.append(temp_vol[i,j,k])

    box_array_np = np.asarray(box_array)
    # print "boxarray", np.shape(box_array_np)
    # print "mean", box_array_np.mean(),box_array_np.max()
    #print np.amin(box_array_np),np.amax(box_array_np), np.mean(box_array_np)
    #mask_vol[vol>(box_array_np.mean() + 0.0* box_array_np.std())] = True
    mask_vol[vol > (0.8*box_array_np.mean() + 0.2*box_array_np.max()) ] = 1
    mask_vol[vol > 100] = 0



    return mask_vol

def Intensity_based_Features_3D(ND, mask_vol):

    Idx_In = np.where(mask_vol == 1)
    Idx_Out = np.where(mask_vol == 0)

    In_arr = []
    Out_arr = []
    #print np.shape(Idx_In)
    for i in range(np.shape(Idx_In)[1]):
        In_arr.append(ND[Idx_In[2][i],Idx_In[1][i],Idx_In[0][i]])
    for j in range(np.shape(Idx_Out)[1]):
        Out_arr.append(ND[Idx_Out[2][j], Idx_Out[1][j], Idx_Out[0][j]])

    InMean = np.mean(In_arr)
    OutMean = np.mean(Out_arr)
    InVar = np.std(In_arr)*np.std(In_arr)
    Kurto = np.mean((In_arr - InMean)**4)/((InVar)**2)
    Skew = np.sum((In_arr-InMean)**3)/((np.shape(Idx_In)[1]-1)**3)


    return InMean, OutMean, InVar, Skew, Kurto

def ND_Analysis_2D(nodule_Img, mask):
    ND_Score = 0
    FeatureVec = []
    if np.amax(mask) >0:
        FeatureVec.append(regionprops(mask,nodule_Img)[0].minor_axis_length/(regionprops(mask)[0].major_axis_length+0.0000000001))
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].area)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].convex_area)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].eccentricity)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].equivalent_diameter)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].extent)
        #FeatureVec.append(regionprops(mask,nodule_Img)[0].filled_area)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[0][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[0][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[1][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor[1][1])
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor_eigvals[0])
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].inertia_tensor_eigvals[1])
        FeatureVec.append(regionprops(mask, nodule_Img)[0].inertia_tensor_eigvals[1] / (regionprops(mask)[0].inertia_tensor_eigvals[0] + 0.0000000001))
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].major_axis_length)
        # FeatureVec.append(regionprops(mask,nodule_Img)[0].minor_axis_length)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].max_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].mean_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].min_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[0][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[1][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][1])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments[2][2])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments_hu[0])
        FeatureVec.append(regionprops(mask,nodule_Img)[0].moments_hu[1])
        FeatureVec.append(regionprops(mask, nodule_Img)[0].orientation)
        FeatureVec.append(regionprops(mask, nodule_Img)[0].perimeter)
        FeatureVec.append(regionprops(mask, nodule_Img)[0].solidity)


        #print FeatureVec


    else:
        FeatureVec = np.zeros(25)

    return FeatureVec

def check_range(coord_X, coord_Y, coord_Z):
    x_max = np.amax(coord_X)
    x_min = np.amin(coord_X)
    y_max = np.amax(coord_Y)
    y_min = np.amin(coord_Y)
    z_max = np.amax(coord_Z)
    z_min = np.amin(coord_Z)

    ##real Bounding Box length
    x_rng = x_max - x_min
    y_rng = y_max - y_min
    z_rng = z_max - z_min

    max_rng = max(max(x_rng, y_rng), z_rng)

    Radius = max_rng/2

    return Radius

def PCA_3D(mask):

    region = regionprops(mask)[0]
    points = []
    coord_X = []
    coord_Y = []
    coord_Z = []
    idx = 0
    for coordinates in region.coords:
        points.append([coordinates[0], coordinates[1], coordinates[2]])
        coord_X.append(coordinates[2])
        coord_Y.append(coordinates[1])
        coord_Z.append(coordinates[0])
        # Input_vol[coord_Z, coord_Y, coord_X] = 255

    data = np.array(points)
    coord_mean = []
    coord_mean.append(np.mean(coord_Z))
    coord_mean.append(np.mean(coord_Y))
    coord_mean.append(np.mean(coord_X))


    R  = check_range(coord_X, coord_Y, coord_Z)

    sklearn_pca = sklearnPCA()
    sklearn_pca.fit(data)
    lamda1 = sklearn_pca.explained_variance_[0] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])
    lamda2 = sklearn_pca.explained_variance_[1] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])
    lamda3 = sklearn_pca.explained_variance_[2] / (
        sklearn_pca.explained_variance_[0] + sklearn_pca.explained_variance_[1] + sklearn_pca.explained_variance_[
            2])

    ELScore = lamda3/(lamda1+0.0000000001)
    CompScore = region.area/((4*np.pi*R*R*R)/3)

    return lamda1,lamda2,lamda3, ELScore, CompScore

def ND_Analysis_3D(nodule_Img, mask):

    FeatureVec = []
    if np.amax(mask) >0:

        FeatureVec.append(regionprops(mask,nodule_Img)[0].extent)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].max_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].mean_intensity)
        FeatureVec.append(regionprops(mask,nodule_Img)[0].min_intensity)

        lamda1, lamda2, lamda3, ELScore, CompScore = PCA_3D(mask)

        FeatureVec.append(lamda1)
        FeatureVec.append(lamda2)
        FeatureVec.append(lamda3)
        FeatureVec.append(ELScore)
        FeatureVec.append(CompScore)

        InMean, OutMean, InVar, Skew, Kurto = Intensity_based_Features_3D(nodule_Img, mask)

        FeatureVec.append(InMean)
        FeatureVec.append(OutMean)
        FeatureVec.append(InVar)
        FeatureVec.append(Skew)
        FeatureVec.append(Kurto)

        #print InMean, OutMean, InVar, Skew, Kurto

        #print FeatureVec


    else:
        FeatureVec = np.zeros(14)

    return FeatureVec

# def Extract_ND_Feature(ND):
#
#     mask_vol = Global_Thresholding_3D(ND)
#
#     nodule_Img1 = ND[int(round(np.shape(ND)[0] / 2)), :, :]
#     nodule_Img2 = ND[:, int(round(np.shape(ND)[0] / 2)), :]
#     nodule_Img3 = ND[:, :, int(round(np.shape(ND)[0] / 2))]
#
#     mask_Img1 = mask_vol[int(round(np.shape(ND)[0] / 2)), :, :]
#     mask_Img2 = mask_vol[:, int(round(np.shape(ND)[0] / 2)), :]
#     mask_Img3 = mask_vol[:, :, int(round(np.shape(ND)[0] / 2))]
#
#     FeatureVec1 = ND_Analysis_2D(nodule_Img1, mask_Img1)
#     FeatureVec2 = ND_Analysis_2D(nodule_Img2, mask_Img2)
#     FeatureVec3 = ND_Analysis_2D(nodule_Img3, mask_Img3)
#     FeatureVec4 = ND_Analysis_3D(ND, mask_vol)
#
#     #ND_Feature = FeatureVec1 + FeatureVec2 + FeatureVec3 + FeatureVec4
#     ND_Feature = np.concatenate((FeatureVec1, FeatureVec2, FeatureVec3, FeatureVec4))
#     #print ND_Feature
#
#     return ND_Feature

def make_diagonal_image(vol, FIX_LEN = 70):
    center = int(FIX_LEN/2)
    rotate = 45
    M = cv2.getRotationMatrix2D((center, center), rotate, 1)
    diagonal_vol = []
    for i in range(0,FIX_LEN):
        diagonal_vol.append(cv2.warpAffine(vol[i], M, (FIX_LEN, FIX_LEN), borderValue=-2000))

    diagonal_vol = np.array(diagonal_vol, dtype=np.float32)
    return np.transpose(diagonal_vol,(1, 2, 0))[35], np.transpose(diagonal_vol,(2,1,0))[35]

def Extract_ND_Feature(ND):

    mask_vol = Global_Thresholding_3D(ND)

    nodule_Img1 = ND[int(round(np.shape(ND)[0] / 2)), :, :]
    nodule_Img2 = ND[:, int(round(np.shape(ND)[0] / 2)), :]
    nodule_Img3 = ND[:, :, int(round(np.shape(ND)[0] / 2))]

    mask_Img1 = mask_vol[int(round(np.shape(ND)[0] / 2)), :, :]
    mask_Img2 = mask_vol[:, int(round(np.shape(ND)[0] / 2)), :]
    mask_Img3 = mask_vol[:, :, int(round(np.shape(ND)[0] / 2))]

    nodule_Img4, nodule_Img5 = make_diagonal_image(ND)
    nodule_Img6, nodule_Img7 = make_diagonal_image(np.transpose(ND, (1, 2, 0)))
    nodule_Img8, nodule_Img9 = make_diagonal_image(np.transpose(ND, (2, 1, 0)))

    temp_mask_vol = np.array(mask_vol, dtype=np.uint8)
    mask_Img4, mask_Img5 = make_diagonal_image(temp_mask_vol)
    mask_Img6, mask_Img7 = make_diagonal_image(np.transpose(temp_mask_vol, (1, 2, 0)))
    mask_Img8, mask_Img9 = make_diagonal_image(np.transpose(temp_mask_vol, (2, 1, 0)))

    mask_Img4 = np.asarray(mask_Img4, dtype=np.uint8, order='C')
    mask_Img5 = np.asarray(mask_Img5, dtype=np.uint8, order='C')
    mask_Img6 = np.asarray(mask_Img6, dtype=np.uint8, order='C')
    mask_Img7 = np.asarray(mask_Img7, dtype=np.uint8, order='C')
    mask_Img8 = np.asarray(mask_Img8, dtype=np.uint8, order='C')
    mask_Img9 = np.asarray(mask_Img9, dtype=np.uint8, order='C')

    FeatureVec1 = ND_Analysis_2D(nodule_Img1, mask_Img1)
    FeatureVec2 = ND_Analysis_2D(nodule_Img2, mask_Img2)
    FeatureVec3 = ND_Analysis_2D(nodule_Img3, mask_Img3)
    FeatureVec4 = ND_Analysis_2D(nodule_Img4, mask_Img4)
    FeatureVec5 = ND_Analysis_2D(nodule_Img5, mask_Img5)
    FeatureVec6 = ND_Analysis_2D(nodule_Img6, mask_Img6)
    FeatureVec7 = ND_Analysis_2D(nodule_Img7, mask_Img7)
    FeatureVec8 = ND_Analysis_2D(nodule_Img8, mask_Img8)
    FeatureVec9 = ND_Analysis_2D(nodule_Img9, mask_Img9)
    FeatureVec10 = ND_Analysis_3D(ND, mask_vol)
    #print (np.shape(FeatureVec1))
    #print (np.shape(FeatureVec2))
    #print (np.shape(FeatureVec3))
    #print (np.shape(FeatureVec4))


    #ND_Feature = FeatureVec1 + FeatureVec2 + FeatureVec3 + FeatureVec4
    ND_Feature = np.concatenate((FeatureVec1, FeatureVec2, FeatureVec3, FeatureVec4, FeatureVec5,
                                 FeatureVec6, FeatureVec7, FeatureVec8, FeatureVec9, FeatureVec10))
    #print ND_Feature

    return ND_Feature

if __name__=="__main__":
    INPUT_FOLDER ='/home/kwanghee/LIDC/dsb2017/KAGGLE_NODULE_CROP_ALL_NPY_MG10/V0/'
    nodules = os.listdir(INPUT_FOLDER)
    nodules.sort()
    print nodules

    ND_Features = []
    for i in xrange(len(nodules)):
        ND = np.load(INPUT_FOLDER + nodules[i])
        ND_Feature = Extract_ND_Feature(ND)
        #ND_Features.append(ND_Feature)
        print (ND_Feature)


        # plt.figure(100)
        # plt.imshow(ND[int(round(np.shape(ND)[0]/2))], cmap=plt.cm.gray)
        # plt.figure(101)
        # #plt.imshow(np.max(vol_rescale, axis=0), cmap=plt.cm.gray)
        # plt.imshow(ND[:,int(round(np.shape(ND)[0]/2)),:], cmap=plt.cm.gray)
        # plt.figure(102)
        # #plt.imshow(np.max(vol_rescale[], axis=0), cmap=plt.cm.gray)
        # plt.imshow(ND[:,:, int(round(np.shape(ND)[0] / 2))], cmap=plt.cm.gray)
        # plt.figure(103)
        # plt.imshow(np.max(ND, axis=0), cmap=plt.cm.gray)
        # plt.show()
        #plot_3d(ND, 0)