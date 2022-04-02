import torch
from torch.utils.data import Dataset
import os 
import numpy as np
import nibabel as nib
from torchvision import transforms as T
import random
import torch.nn.functional as F
import nrrd
import pandas as pd
from torchio import Subject, ScalarImage, RandomAffine
import scipy
import sys


def NORM(img):
    img_out = np.zeros(img.shape)
    Max = np.max(img)
    Min = np.min(img)
    out = (img-Min)/(Max-Min+1e-10)
    img_out=out
    return img_out
    
def GNORM(img):
    img_out = np.zeros(img.shape)
    Mean = np.mean(img)
    Std = np.std(img)
    out = (img-Mean)/(Std+1e-10)
    img_out=out
    return img_out
class US_Dataset(Dataset):
    def __init__(self, path_us_3d,path_us_2d,path_us_3d_deformed,path_params,transforms = None):
        super().__init__()
        self.path_us_3d = path_us_3d
        self.path_us_2d = path_us_2d
        self.path_us_3d_deformed = path_us_3d_deformed
        self.path_params = path_params
        
        
        self.data_list_us_3d = sorted(os.listdir(path_us_3d))
        self.data_list_us_2d = sorted(os.listdir(path_us_2d))
        self.data_list_us_3d_deformed = sorted(os.listdir(path_us_3d_deformed))
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data_list_us_3d)

    def __getitem__(self,item):
        
        params = pd.read_csv(self.path_params)
        new_name,_,_ = self.data_list_us_3d[item].split(sep='.')
        #_,name_new = name.split(sep='_')
        #print(name_new)
        us_img_3d = nib.load(self.path_us_3d+self.data_list_us_3d[item]).get_fdata().astype('float64')
        #us_img_2d = nib.load(self.path_us_2d+self.data_list_us_2d[item]).get_fdata().astype('float64')
        us_img_3d_deformed = nib.load(self.path_us_3d_deformed+self.data_list_us_3d_deformed[item]).get_fdata().astype('float64')
        us_img_2d = us_img_3d_deformed[5,:,:]
        #print('The shape of the 3d orig',us_img_3d.shape)
        #print('The shape of the 3d deformed',us_img_3d_deformed.shape)
        #print('The shape of the 2d deformed',us_img_2d.shape)
        
        Rot_x = torch.tensor(params['Rot_x'].loc[item])
        Rot_y = torch.tensor(params['Rot_y'].loc[item])
        Rot_z = torch.tensor(params['Rot_z'].loc[item])
        Shift_x = torch.tensor(params['Shift_x'].loc[item])
        Shift_y = torch.tensor(params['Shift_y'].loc[item])
        Shift_z = torch.tensor(params['Shift_z'].loc[item])
        
        
        us_img_3d = NORM(us_img_3d)
        us_img_2d = NORM(us_img_2d)
        us_img_3d_deformed = NORM(us_img_3d_deformed)
           
        
        if self.transforms is not None:
            us_img_3d = self.transforms(us_img_3d).type(torch.FloatTensor).unsqueeze(0)
            
            #us_img_2d = us_img_2d.squeeze(2)
            us_img_2d = self.transforms(us_img_2d).type(torch.FloatTensor)#.unsqueeze(0)
            #us_img_3d_deformed = us_img_3d_deformed.squeeze(1)
            #print('The shape of us_img_3d_deformed',us_img_3d_deformed.shape)
            us_img_3d_deformed = self.transforms(us_img_3d_deformed).type(torch.FloatTensor).unsqueeze(0)
        #print('The shape of the 3d orig',us_img_3d.shape)
        #print('The shape of the 3d deformed',us_img_3d_deformed.shape)
        #print('The shape of the 2d deformed',us_img_2d.shape)
        #us_img_3d = torch.transpose(us_img_3d, 1, 2)   
        #us_img_3d_deformed = torch.transpose(us_img_3d_deformed, 1, 2) 
        us_img_3d = torch.transpose(us_img_3d, 2, 3)   
        us_img_3d_deformed = torch.transpose(us_img_3d_deformed, 2, 3) 
        us_img_3d = torch.transpose(us_img_3d, 1, 3)   
        us_img_3d_deformed = torch.transpose(us_img_3d_deformed, 1, 3) 
        #print('The shape of the 3d orig after',us_img_3d.shape)
        #print('The shape of the 3d deformed after',us_img_3d_deformed.shape)
        #print('The shape of the 2d deformed after',us_img_2d.shape)
        #sys.exit()  
        Rot_y = 0.0
        Rot_z = 0.0       
        return us_img_3d, us_img_2d, us_img_3d_deformed, Rot_x, Rot_y,Rot_z,Shift_x,Shift_y,Shift_z,new_name
      
