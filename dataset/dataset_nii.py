import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .rand import Uniform
from .transforms import (CenterCrop, Compose, Flip, GaussianBlur, Identity,
                         Noise, Normalize, NumpyType, Pad, RandCrop,
                         RandCrop3D, RandomFlip, RandomIntensityChange,
                         RandomRotion, RandSelect, Rot90)


mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]])


class BraTS_load_all_train_nii_class(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.xlsx'):
        data_file_path = os.path.join(root, train_file)
        df = pd.read_excel(data_file_path)

        self.names = df["patient_id"].tolist()
        # self.class_labels = df["WHO"].tolist()
        self.class_labels = df["IDH"].tolist()

        self.volpaths = [os.path.join(root, 'vol', name + '_vol.npy') for name in self.names]
        self.transforms = eval(transforms or 'Identity()')
        self.num_cls = num_cls

        self.modal_ind = {
            'flair': np.array([0]),
            't1ce': np.array([1]),
            't1': np.array([2]),
            't2': np.array([3]),
            'all': np.array([0, 1, 2, 3])
        }.get(modal, np.array([0, 1, 2, 3]))

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]
        class_label = self.class_labels[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))

        _, H, W, Z = np.shape(y)
        one_hot_target = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_target, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        class_label = torch.tensor(class_label, dtype=torch.long)
        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        return x, yo, class_label, mask, name

    def __len__(self):
        return len(self.volpaths)



class Brats_load_all_test_nii_class(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.xlsx'):
        data_file_path = os.path.join(root, test_file)
        df = pd.read_excel(data_file_path)

        self.names = df["patient_id"].tolist()
        # self.class_labels = df["class_who"].tolist()
        self.class_labels = df["is_IDH_muted"].tolist()

        self.volpaths = [os.path.join(root, 'vol', name + '_vol.npy') for name in self.names]
        self.transforms = eval(transforms or 'Identity()')

        self.modal_ind = {
            'flair': np.array([0]),
            't1ce': np.array([1]),
            't1': np.array([2]),
            't2': np.array([3]),
            'all': np.array([0, 1, 2, 3])
        }.get(modal, np.array([0, 1, 2, 3]))

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        class_label = self.class_labels[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, class_label, name

    def __len__(self):
        return len(self.volpaths)



class BraTS_load_all_train_nii_segmentation(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.xlsx'):
        data_file_path = os.path.join(root, train_file)
        df = pd.read_excel(data_file_path)

        self.names = df["patient_id"].tolist()
        self.volpaths = [os.path.join(root, 'vol', name + '_vol.npy') for name in self.names]

        self.transforms = eval(transforms or 'Identity()')
        self.num_cls = num_cls

        self.modal_ind = {
            'flair': np.array([0]),
            't1ce': np.array([1]),
            't1': np.array([2]),
            't2': np.array([3]),
            'all': np.array([0, 1, 2, 3])
        }.get(modal, np.array([0, 1, 2, 3]))

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))

        _, H, W, Z = np.shape(y)
        one_hot_target = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_target, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)



class BraTS_load_all_test_nii_segmentation(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.xlsx'):
        data_file_path = os.path.join(root, test_file)
        df = pd.read_excel(data_file_path)

        self.names = df["patient_id"].tolist()
        self.volpaths = [os.path.join(root, 'vol', name + '_vol.npy') for name in self.names]

        self.transforms = eval(transforms or 'Identity()')

        self.modal_ind = {
            'flair': np.array([0]),
            't1ce': np.array([1]),
            't1': np.array([2]),
            't2': np.array([3]),
            'all': np.array([0, 1, 2, 3])
        }.get(modal, np.array([0, 1, 2, 3]))

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)


if __name__ == "__main__":
    excel_path = ""  # Excel_path
    data_root = ""  # data_path
    batch_size = 2

    dataset = BraTS_load_all_train_nii_class(
        transforms='Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])',  # 不使用数据增强
        root=data_root,
        modal='all',
        num_cls=4,
        train_file=excel_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    sample = next(iter(dataloader))
    print("Image shape:", sample[0].shape)
    print("Segmentation Label shape:", sample[1].shape)
    print("Class Label:", sample[2])
    print("mask:", sample[3])
    print("Filename:", sample[4])

    excel_path1 = ""  # Excel_path

    test_dataset = Brats_load_all_test_nii_class(
        transforms='Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])',
        root=data_root,
        modal='all',
        test_file=excel_path1
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    sample1 = next(iter(test_dataloader))
    print("Image shape:", sample1[0].shape)
    print("Segmentation Label shape:", sample1[1].shape)
    print("Class Label:", sample1[2])
    print("Filename:", sample1[3])
