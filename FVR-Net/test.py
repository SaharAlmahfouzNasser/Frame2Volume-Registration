import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import dataloader_test
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image
import random
from monai.losses import LocalNormalizedCrossCorrelationLoss,DiceFocalLoss, BendingEnergyLoss,GlobalMutualInformationLoss
from monai.networks.layers import Norm
from option_test import args
from train import *
#from losses import *
from networks.fvrnet import mynet3
from torch.nn import init
import time

TEST_BATCH_SIZE = args.TEST_BATCH_SIZE

WORKERS = args.WORKERS

device1 = args.DEVICE1
device2 = args.DEVICE2
device3 = args.DEVICE3

TEST_US_3D = args.TEST_US_3D

TEST_US_3D_DEFORMED = args.TEST_US_3D_DEFORMED

TEST_US_2D = args.TEST_US_2D

PARAMS = args.PARAMS

LOAD_CHECKPOINT = args.LOAD_CHECKPOINT

EPOCHS = args.EPOCHS

SAVE_PATH = args.SAVE_PATH

def pbar_desc(label, epoch, total_epochs, parameters_loss, similarity_loss,smoothness_loss,total_loss):
    return f'{label}:{epoch:04d}/{total_epochs} | {parameters_loss: .3f} | {similarity_loss: .3f} | {smoothness_loss: .3f} | {total_loss: .3f}'


def test(net, tst_dl,epoch,epochs,loss1,loss2,loss3,device1,device2,device3,SAVE_PATH):


    t_pbar = tqdm(tst_dl, desc=pbar_desc('test',epoch,epochs,0.0,0.0,0.0,0.0))


    net.eval()

    avg_loss = []
    PARAM_LOSS =[]

    with torch.no_grad():
        no = 0

        for us_img_3d, us_img_2d, us_img_3d_deformable,Rot_x,Rot_y,Rot_z,Shift_x,Shift_y,Shift_z,name in t_pbar:

            no +=1
            us_img_3d = us_img_3d.to(device2)#.squeeze(1)
            us_img_3d_deformable = us_img_3d_deformable.to(device2)#.squeeze(1)
            us_img_2d = us_img_2d.to(device2).squeeze(3)
            Rot_x = Rot_x.to(device2)
            Rot_y = Rot_y.to(device2)
            Rot_z = Rot_z.to(device2)
            Shift_x = Shift_x.to(device2)
            Shift_y = Shift_y.to(device2)
            Shift_z = Shift_z.to(device2)

            vol_resampled, deformation_params = net(us_img_3d, us_img_2d, device = device2)
            deformation_params = torch.transpose(deformation_params,0,1)
            Loss_params =torch.mean(loss1(deformation_params[3],Rot_x)+loss1(deformation_params[4],Rot_y)
                                    + loss1(deformation_params[5],Rot_z)+loss1(deformation_params[0],Shift_x)
                                    + loss1(deformation_params[1],Shift_y)+loss1(deformation_params[2],Shift_z))

            ## add smoothness as well over the deformation field
            smoothness_loss = 0.0
            Loss_similarity = loss3(vol_resampled[0,0,5,:,:], us_img_3d_deformable[0,0,5,:,:])
            total_loss = Loss_params + Loss_similarity
            total_loss = torch.tensor(total_loss,dtype=torch.float64)

            t_pbar.set_description(pbar_desc('val',epoch,epochs,Loss_similarity.item(),Loss_params.item(),smoothness_loss,total_loss.item()))


            vol_resampled =vol_resampled.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_3d = us_img_3d.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_2d = us_img_2d.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_3d_deformable = us_img_3d_deformable.detach().cpu().squeeze(0).squeeze(0).numpy()
            deformation_params = deformation_params.detach().cpu().squeeze(0).squeeze(0).numpy()

            
                
            vol_resampled_new = nib.Nifti1Image(vol_resampled, np.eye(4))
            vol_resampled_new.header.get_xyzt_units()
            vol_resampled_new.to_filename('./'+SAVE_PATH+'/vol_resampled_'+name[0]+'.nii.gz')

            us_img_3d_new  = nib.Nifti1Image(us_img_3d, np.eye(4))
            us_img_3d_new.header.get_xyzt_units()
            us_img_3d_new.to_filename('./'+SAVE_PATH+'/us_img_3d_'+name[0]+'.nii.gz')

            us_img_2d_new = nib.Nifti1Image(us_img_2d, np.eye(4))
            us_img_2d_new.header.get_xyzt_units()
            us_img_2d_new.to_filename('./'+SAVE_PATH+'/us_img_2d_'+name[0]+'.nii.gz')

            us_img_3d_deformable_new = nib.Nifti1Image(us_img_3d_deformable, np.eye(4))
            us_img_3d_deformable_new.header.get_xyzt_units()
            us_img_3d_deformable_new.to_filename('./'+SAVE_PATH+'/us_img_3d_deformable_'+name[0]+'.nii.gz')

                #### TO DO ####
                #### SAVE THE PREDICTED PARAMETERS ##########


            avg_loss.append(total_loss.item())
            PARAM_LOSS.append(Loss_params.item())



        avg_loss = torch.mean(torch.tensor(avg_loss))



        avg_loss = torch.mean(torch.tensor(avg_loss))

        avg_param_loss = torch.mean(torch.tensor(PARAM_LOSS))
        std_param_loss = torch.std(torch.tensor(PARAM_LOSS))


        
    return torch.tensor(PARAM_LOSS), avg_param_loss, std_param_loss


def main():
    
    transforms = T.Compose([T.ToTensor()])
    tst_ds = dataloader_test.US_Dataset(TEST_US_3D,TEST_US_2D,TEST_US_3D_DEFORMED,PARAMS,transforms)
    tst_dl = DataLoader(tst_ds, TEST_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    transforms = T.Compose([T.ToTensor()])
    start_epoch = 1
    layers = [3, 8, 36, 3]
    net = mynet3(layers)
    


    print('Net parameters:', sum(p.numel() for p in net.parameters()))

    loss1 = torch.nn.L1Loss().to(device2)#torch.nn.MSELoss().to(device2)
    loss2 = BendingEnergyLoss().to(device2)
    loss3 =torch.nn.MSELoss().to(device2)# LocalNormalizedCrossCorrelationLoss().to(device2)#GlobalMutualInformationLoss().to(device2)# DiceFocalLoss ().to(device3)#torch.nn.LogSoftmax().to(device3)
    


    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module = dill,map_location=device2)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net_state_dict'],strict=False)
        opt = checkpoint['optimizer']
        sched = checkpoint['lr_scheduler']


    net.to(device2)

    

    

   
    #### for mixed precision ####
    #scaler = GradScaler()
    start_time = time.time()
    #for epoch in range(start_epoch, EPOCHS+1):
    print('Testing in progress.....')
    PATAM_LOSS, avg_param_loss, std_param_loss = test(net, tst_dl,start_epoch,EPOCHS,loss1,loss2,loss3,device1,device2,device3,SAVE_PATH)
    print("Testing Results are: mean, std", avg_param_loss, std_param_loss)
    Time = time.time()-start_time
    print('Total running time is', Time) 

if __name__=='__main__':
    
    main()
