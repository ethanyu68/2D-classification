import cv2
import numpy as np
import os, glob
import h5py
import argparse
import matplotlib.pyplot as plt
import pdb
import pandas as pd

parser = argparse.ArgumentParser(description='Generate H5 file and augmentation')
parser.add_argument("--path_IMG",type=str,default = './PNG/IMG')
parser.add_argument("--path_infotable",type=str,default = 'PIH_Variables.csv')

parser.add_argument("--path_write_h5",type=str, default = 'H5/CT_2D_IMG_MASK_PIH_variables.h5')
parser.add_argument("--interval",type=int, default=2, help="interval between two slices included in cube")

opt = parser.parse_args()

path_dataset = opt.path_IMG
path_infotable = opt.path_infotable # path of csv file of information table

interval = opt.interval
dim_img = 512

dir_data = glob.glob(path_dataset + '/*')  # Get list of directories of NPIH
infotable = pd.read_csv(path_infotable)

num_total_slices = (len(dir_data) - 4)*20
num_total_patient = (len(dir_data) - 4)
# data
dataset = np.zeros([num_total_slices, 1, dim_img, dim_img])

# label
PIH_NPIH = np.zeros([num_total_patient, 1])
CMV = np.zeros([num_total_patient, 1])
paeni = np.zeros([num_total_patient, 1])

ID_slice = []
ID_patient = []

tooshort = 0
slice = 0
patient = 0
for path_folder in dir_data:
    ID = int(path_folder.split('/')[-1][:4])
    ID_patient.append(ID)
    ########################################################################################################################
    # label
    loc_in_table = np.where(infotable['StudyID'] == ID)[0][0]
    if infotable['Hydrocephalus'][loc_in_table] == 'PIH':
        PIH_NPIH[patient] = 1
    else:
        PIH_NPIH[patient] = 0
    CMV[patient] = infotable['CMVCSF'][loc_in_table]
    paeni[patient] = infotable['qpcr_genus_9_28_21'].to_numpy()[np.where(infotable['StudyID'].to_numpy() == ID)[0][0]]
    # info
    LD[patient][infotable['LD'][loc_in_table]-1] = 1
    ## region
    if infotable['Region'][loc_in_table] == 'Central':
        region[patient][0] = 1
    elif infotable['Region'][loc_in_table] == 'Northern':
        region[patient][1] = 1
    elif infotable['Region'][loc_in_table] == 'Western':
        region[patient][2] = 1
    else:
        region[patient][3] = 1
    ########################################################################################################################
    # fetch list of png files
    dir_img = sorted(glob.glob(path_folder+'/*.png'))
    # remove folders containing too few images
    length_slices = len(dir_img)
    if length_slices < 13 :
        print('>>> Numbers of slices in' + path_folder + ' is ' + str(length_slices))
    if length_slices > 50 :
        print('>>> Numbers of slices in' + path_folder + ' is ' + str(length_slices))
    idx_start = int(length_slices*0.33)
    idx_end = int(length_slices*0.67) # Set starting index of slice from middle area
    # read and save images and masks
    for j in range(idx_start, idx_end, interval):
        # remoteCT scan
        img = cv2.imread(dir_img[j])
        img = img[:, :, 0].astype(np.float32)/255
        dataset[slice, 0, :, :] = img
        # brain mask
        tmp = dir_img[j].split('/')
        path_mask = os.path.join(path_maskset, tmp[-2], tmp[-1])
        mask = cv2.imread(path_mask)
        mask = mask[:, :, 0].astype(np.float32)/255
        maskset[slice, 0, :, :] = mask

        slice += 1
        ID_slice.append(ID)
    patient = patient + 1
num_slice = slice
dataset = dataset[:num_slice]
maskset = maskset[:num_slice]
PIH_NPIH = PIH_NPIH[:num_slice]
CMV = CMV[: num_slice]
paeni = paeni[:num_slice]
region = region[:num_slice]
LD = LD[: num_slice]
ID_slice = np.array(ID_slice)
ID_patient = np.array(ID_patient)
# save all data
h5w = h5py.File(opt.path_write_h5, 'w')
h5w.create_dataset(name='data', dtype=np.float32, shape=dataset.shape, data=dataset)
h5w.create_dataset(name='mask', dtype=np.float32, shape=maskset.shape, data=maskset)
h5w.create_dataset(name='hydrocephalus', dtype=int, shape=PIH_NPIH.shape, data=PIH_NPIH)
h5w.create_dataset(name='CMV', dtype=int, data=CMV)
h5w.create_dataset(name='paeni', dtype=int, data=paeni)
h5w.create_dataset(name='region', dtype=np.float32, data=region)
h5w.create_dataset(name='LD', dtype=np.float32, data=LD)
h5w.create_dataset(name='ID_slice', dtype=int, data=ID_slice)
h5w.create_dataset(name='ID_patient', dtype=int, data=ID_patient)

h5w.close()
















