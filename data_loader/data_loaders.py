import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class LozengeTrainDataset(Dataset):
    def __init__(self):
        self.images_train = np.load('data_loader/data/red_lozenges/images_train.npy')
        self.centers_train = np.load('data_loader/data/red_lozenges/centers_train.npy')
        self.scales_train = np.load('data_loader/data/red_lozenges/scales_train.npy')
        self.rots_train = np.load('data_loader/data/red_lozenges/rots_train.npy')

    def __len__(self):
        return self.images_train.shape[0]

    def __getitem__(self,idx):
        img = torch.from_numpy(self.images_train[idx]).float()
        center = torch.from_numpy(self.centers_train[idx]).float()
        scale = torch.tensor(self.scales_train[idx]).float()
        rot = torch.tensor(self.rots_train[idx]).float()

        ## setting from HxWxC to CxHxW
        img = np.transpose(img,(2,0,1))

        target = (center[0],center[1],scale,rot)
        target = torch.tensor(target)
        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (img,target)

class LozengeTrainLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = LozengeTrainDataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class LozengeTestDataset(Dataset):
    def __init__(self,transform=None):
        self.images_test = np.load('data_loader/data/red_lozenges/images_test.npy')
        self.centers_test = np.load('data_loader/data/red_lozenges/centers_test.npy')
        self.scales_test = np.load('data_loader/data/red_lozenges/scales_test.npy')
        self.rots_test = np.load('data_loader/data/red_lozenges/rots_test.npy')

    def __len__(self):
        return self.images_test.shape[0]

    def __getitem__(self,idx):
        img = torch.from_numpy(self.images_test[idx]).float()
        center = torch.from_numpy(self.centers_test[idx]).float()
        scale = torch.tensor(self.scales_test[idx]).float()
        rot = torch.tensor(self.rots_test[idx]).float()

        img = np.transpose(img,(2,0,1))
        target = (center[0],center[1],scale,rot)
        target = torch.tensor(target)
        #Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (img,target)

class LozengeTestLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = LozengeTestDataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class deltaLozengeTrainDataset(Dataset):
    def __init__(self):
        self.images_train = np.load('data_loader/data/red_lozenges/images_train.npy')
        self.centers_train = np.load('data_loader/data/red_lozenges/centers_train.npy')
        self.scales_train = np.load('data_loader/data/red_lozenges/scales_train.npy')
        self.rots_train = np.load('data_loader/data/red_lozenges/rots_train.npy')

        self.ref_image = torch.tensor(np.transpose(self.images_train[0],(2,0,1)))
        self.ref_label = torch.tensor((self.centers_train[0,0],self.centers_train[0,1],self.scales_train[0],self.rots_train[0]))

    def __len__(self):
        return self.images_train.shape[0]

    def __getitem__(self,idx):
        ref_img = torch.from_numpy(self.images_train[idx]).float()
        ref_center = torch.from_numpy(self.centers_train[idx]).float()
        ref_scale = torch.tensor(self.scales_train[idx]).float()
        ref_rot = torch.tensor(self.rots_train[idx]).float()

        ## setting from HxWxC to CxHxW
        ref_img = np.transpose(ref_img,(2,0,1))

        ref_target = (ref_center[0],ref_center[1],ref_scale,ref_rot)
        ref_target = torch.tensor(ref_target)

        _idx = np.random.randint(0,self.__len__())
        
        trans_img = torch.from_numpy(self.images_train[_idx]).float()
        trans_center = torch.from_numpy(self.centers_train[_idx]).float()
        trans_scale = torch.tensor(self.scales_train[_idx]).float()
        trans_rot = torch.tensor(self.rots_train[_idx]).float()

        trans_img = np.transpose(trans_img,(2,0,1))

        trans_target = (trans_center[0],trans_center[1],trans_scale,trans_rot)
        trans_target = torch.tensor(trans_target)
            
        delta_target = trans_target - ref_target

        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (ref_img,trans_img,delta_target)

class deltaLozengeTrainLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = deltaLozengeTrainDataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class deltaLozengeTestDataset(Dataset):
    def __init__(self,transform=None):
        self.images_test = np.load('data_loader/data/red_lozenges/images_test.npy')
        self.centers_test = np.load('data_loader/data/red_lozenges/centers_test.npy')
        self.scales_test = np.load('data_loader/data/red_lozenges/scales_test.npy')
        self.rots_test = np.load('data_loader/data/red_lozenges/rots_test.npy')

    def __len__(self):
        return self.images_test.shape[0]

    def __getitem__(self,idx):
        ref_img = torch.from_numpy(self.images_test[idx]).float()
        ref_center = torch.from_numpy(self.centers_test[idx]).float()
        ref_scale = torch.tensor(self.scales_test[idx]).float()
        ref_rot = torch.tensor(self.rots_test[idx]).float()

        ## setting from HxWxC to CxHxW
        ref_img = np.transpose(ref_img,(2,0,1))

        ref_target = (ref_center[0],ref_center[1],ref_scale,ref_rot)
        ref_target = torch.tensor(ref_target)

        _idx = np.random.randint(0,self.__len__())
        
        trans_img = torch.from_numpy(self.images_test[_idx]).float()
        trans_center = torch.from_numpy(self.centers_test[_idx]).float()
        trans_scale = torch.tensor(self.scales_test[_idx]).float()
        trans_rot = torch.tensor(self.rots_test[_idx]).float()

        trans_img = np.transpose(trans_img,(2,0,1))

        trans_target = (trans_center[0],trans_center[1],trans_scale,trans_rot)
        trans_target = torch.tensor(trans_target)
            
        delta_target = trans_target - ref_target

        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (ref_img,trans_img,delta_target)

class deltaLozengeTestLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = deltaLozengeTestDataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

