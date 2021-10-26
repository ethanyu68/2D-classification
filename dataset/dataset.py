import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
from skimage.transform import rotate
import scipy.ndimage as ndimage
import torch
import skimage.measure as measure
import matplotlib.pyplot as plt


class DatasetFromHdf5(data.Dataset):

    def __init__(self, file_path, augmentation=1):
        super(DatasetFromHdf5, self).__init__()
        self.aug = augmentation
        hf = h5py.File(file_path, 'r')
        self.data = hf.get('data') # N x D x H x W
        self.label = hf.get('label')
        self.num_sub = self.data.shape[0]
        self.num_img_per_sub = self.data.shape[1]
        self.num_imgs = self.num_sub * self.num_img_per_sub

    def __getitem__(self, index):
        index_sub = index // self.num_img_per_sub
        index_img = index % self.num_img_per_sub
        img = self.data[index_sub, index_img, :, :] # fetch the img from subject <index_sub> and the <index_img>th img
        label = self.label[index_sub]
        # ======# augmentation # ========================================================================================================================
        # rotation
        if self.aug == 1:
            r = np.random.randint(2)
            if r == 1:
                r1 = np.random.randint(-10, 10)
                img = rotate(img, angle=2 * r1, mode='wrap')

            # cropping and resizing
            r = np.random.randint(6)
            H, W = img.shape
            if r == 1:
                img = cv2.resize(img[int(0.1 * H): int(0.9 * H), int(0.1 * W): int(0.9 * W)], (H, W))
            elif r == 2:
                img = cv2.resize(img[int(0.2 * H): int(0.8 * H), int(0.2 * W): int(0.8 * W)], (H, W))
            else:
                img = img

            # flipping
            r = np.random.randint(6)
            if r == 0:
                img = np.flip(img, axis=1)

            elif r == 1:
                img = np.flip(img, axis=0)

            # noise
            r = np.random.randint(6)
            if r == 1:
                img = np.random.normal(0, 0.015, [H, W]) + img
            elif r == 2:
                img = np.random.normal(0, 0.01, [H, W]) + img
            img = img[np.newaxis, :, :]

        else:
            img = img[np.newaxis, :, :]

        return torch.from_numpy(img.copy()).float(), \
               torch.from_numpy(label).long()


    def __len__(self):
        return self.num_img_per_sub
