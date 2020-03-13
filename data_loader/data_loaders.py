import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np
import imageio as iio
from imgaug.augmenters.geometric import Affine



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
        self.ref_image = dataset.ref_image.unsqueeze_(0).float()
        self.ref_label = dataset.ref_label.unsqueeze_(0).float()
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
            
        #delta_target = trans_target - ref_target

        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (ref_img,trans_img,ref_target)

class deltaLozengeTestLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = deltaLozengeTestDataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class deltaRefLozengeTrainDataset(Dataset):
    def __init__(self):
        self.images_train = np.load('data_loader/data/red_lozenges/images_train.npy')
        self.centers_train = np.load('data_loader/data/red_lozenges/centers_train.npy')
        self.scales_train = np.load('data_loader/data/red_lozenges/scales_train.npy')
        self.rots_train = np.load('data_loader/data/red_lozenges/rots_train.npy')

        self.ref_image = torch.tensor(np.transpose(self.images_train[0],(2,0,1)))
        self.ref_label = torch.tensor([self.centers_train[0,0],self.centers_train[0,1],self.scales_train[0],self.rots_train[0]])

        self.center = self.ref_label[0:2].numpy()
        self.scale = self.ref_label[2].numpy()
        self.rot = self.ref_label[3].numpy()
    def __len__(self):
        return self.images_train.shape[0]

    def __getitem__(self,idx):
        ref_img = self.ref_image
        ref_center = self.center
        ref_scale = self.scale
        ref_rot = self.rot 

        _idx = np.random.randint(0,self.__len__())
        
        trans_img = torch.from_numpy(self.images_train[_idx]).float()
        trans_center = self.centers_train[_idx]
        trans_scale = self.scales_train[_idx]
        trans_rot = self.rots_train[_idx]

        trans_img = np.transpose(trans_img,(2,0,1))
        
        center = np.array([32,32])

        ## From the reference paper
        lambda_target = trans_scale/ref_scale
        rots_target = trans_rot - ref_rot
        translation_target = trans_center - (lambda_target*np.dot(np.array([[np.cos(rots_target),-np.sin(rots_target)],[np.sin(rots_target),np.cos(rots_target)]]),ref_center - center)) + center

        delta_target = torch.tensor([translation_target[0],translation_target[1],lambda_target,np.sin(rots_target)])
        
        return (ref_img.squeeze(0).float(),trans_img.float(),delta_target.float())

class deltaRefLozengeTrainLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = deltaRefLozengeTrainDataset()
        self.ref_image = dataset.ref_image.unsqueeze_(0).float()
        self.ref_label = dataset.ref_label.unsqueeze_(0).float()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class deltaRefLozengeTestDataset(Dataset):
    def __init__(self,transform=None):
        self.images_train = np.load('data_loader/data/red_lozenges/images_train.npy')[0]
        self.centers_train = np.load('data_loader/data/red_lozenges/centers_train.npy')[0]
        self.scales_train = np.load('data_loader/data/red_lozenges/scales_train.npy')[0]
        self.rots_train = np.load('data_loader/data/red_lozenges/rots_train.npy')[0]

        ## For use in the deltaRefTrainer._valid_epoch()
        self.ref_image = torch.tensor(np.transpose(self.images_train,(2,0,1)))
        self.ref_label = torch.tensor((self.centers_train[0],self.centers_train[1],self.scales_train,self.rots_train))

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

           
        #delta_target = trans_target - ref_target

        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (ref_img,ref_target)

class deltaRefLozengeTestLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = deltaRefLozengeTestDataset()
        self.ref_image = dataset.ref_image.unsqueeze_(0).float()
        self.ref_label = dataset.ref_label.unsqueeze_(0).float()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class imgTrainDataset(Dataset):
    def __init__(self,length = 10000):
        self.img = iio.imread('data_loader/data/rabbit_small.jpg')/255.

        # Bounding box coordinates x1, y1, x2, y2
        self.box = [18,18,42,42]
        self.len = length

        self.centers = np.random.randint(-12, 12, length)
        self.rots = np.random.rand(length)*np.pi/2
        self.scales = (np.random.rand(length)-0.5)/5. + 1.

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        img = self.img

        ## setting from HxWxC to CxHxW
        transform = Affine(translate_px=self.centers[idx],rotate=self.rots[idx]*180,scale=self.scales[idx])
        img = transform.augment_image(img)
        img = np.transpose(img,(2,0,1))

        target = (self.centers[idx],self.centers[idx],self.scales[idx],self.rots[idx])
        target = torch.tensor(target)
        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (torch.tensor(np.transpose(self.img,(2,0,1))).float(),torch.tensor(img).float(),target)

class imgTrainLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = imgTrainDataset()
        self.ref_image = torch.tensor(np.transpose(dataset.img,(2,0,1))).float()
        self.ref_label = torch.tensor([0.0,0.0,0.0,0.0]).float()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

class imgTestDataset(Dataset):
    def __init__(self,length = 1000):
        self.img = iio.imread('data_loader/data/rabbit_small.jpg')/255.

        # Bounding box coordinates x1, y1, x2, y2
        self.box = [18,18,42,42]
        self.len = length

        self.centers = np.random.randint(-12, 12, length)
        self.rots = np.random.rand(length)*np.pi
        self.scales = (np.random.rand(length)-0.5)/5. + 1.

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        img = self.img

        ## setting from HxWxC to CxHxW
        transform = Affine(translate_px=self.centers[idx],rotate=self.rots[idx]*180,scale=self.scales[idx])
        img = transform.augment_image(img)
        img = np.transpose(img,(2,0,1))

        target = (self.centers[idx],self.centers[idx],self.scales[idx],self.rots[idx])
        target = torch.tensor(target)
        ## Not putting transform because the labels can get messed up. Do data augmentation in the dataset generation instead pls
        return (torch.tensor(np.transpose(self.img,(2,0,1))).float(),torch.tensor(img).float(),target)

class imgTestLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = imgTrainDataset()
        self.ref_image = torch.tensor(np.transpose(dataset.img,(2,0,1))).float()
        self.ref_label = torch.tensor([0.0,0.0,0.0,0.0]).float()
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)


