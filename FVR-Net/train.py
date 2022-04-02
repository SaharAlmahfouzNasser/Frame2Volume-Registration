###### Training/Validation ################

import numpy as np
import os
import torch
import dill
from tqdm import tqdm
import random
from torch import nn
from tensorboardX import SummaryWriter
import monai
from monai.losses import BendingEnergyLoss,LocalNormalizedCrossCorrelationLoss
import torch.nn.functional as F
import PIL
from PIL import Image
#from losses import LOG,SurfaceLoss
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler
import pdb
from torch.autograd import Variable

def JacboianDet(pred):
    d1 = int((155-120)/2)
    d2=int((240-192)/2)
    d3=int((240-192)/2)
    J = pred 
    #print("The shape of the deformation feild is:", J.shape)
    
    dy = J[:,:, 1:, :-1] - J[:,:, :-1, :-1]
    dx = J[:,:, :-1, 1:] - J[:,:, :-1, :-1]
    
    Jdet0 = dx[:,0,:,:] * (dy[:,1,:,:])
    Jdet1 = dx[:,1,:,:] * (dy[:,0,:,:])
    
    Jdet = Jdet0 - Jdet1
    #Pad_func = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)
    #Jdet = Pad_func(Jdet)
    
    #Jdet = Jdet.squeeze(0)
    
    #Pad_func2 = nn.ConstantPad3d((d2, 240-(d2+192), d3, 240-(d3+192),d1, 155-(d1+120)), 0)
    #Jdet2 = Pad_func2(Jdet)
    #print("Jdet is of shape:", Jdet2.shape)
    return Jdet


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.BCE = nn.BCEWithLogitsLoss()
        #self.sigmoid = nn.Sigmoid()
    def forward(self, inputs, targets):
        #print("inputs max",torch.max(inputs))
        #inputs = self.sigmoid(inputs) 
        #print(inputs.shape)
        #print(targets.shape)
        BCE_loss = self.BCE(inputs,targets)
        n_BCE_loss = -BCE_loss
        pt = torch.exp(n_BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        #print("FOCALLOSS",F_loss)
        return F_loss


def pbar_desc(label, epoch, total_epochs, parameters_loss, similarity_loss,Gradient,total_loss):
    return f'{label}:{epoch:04d}/{total_epochs} | {parameters_loss: .3f} | {similarity_loss: .3f} | {Gradient: .3f} | {total_loss: .3f}'

def train(net, trn_dl,epoch,epochs,loss1,loss2,loss3,opt,train_losses,device1,device2,device3,TENSORBOARD_LOGDIR,LOSS_WEIGHT,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO):
    
    net.train()
    
    t_pbar = tqdm(trn_dl, desc=pbar_desc('train',epoch,epochs,0.0,0.0,0.0,0.0))
    
    avg_loss = []
    GRAD = []
    for us_img_3d, us_img_2d, us_img_3d_deformable,Rot_x,Rot_y,Rot_z,Shift_x,Shift_y,Shift_z,name in t_pbar:
       
        opt.zero_grad()
        us_img_3d = us_img_3d.to(device2)#.squeeze(1)
        us_img_3d_deformable = us_img_3d_deformable.to(device2)#.squeeze(1)
        us_img_2d = us_img_2d.to(device2).squeeze(3)
        Rot_x = Rot_x.to(device2)
        Rot_y = Rot_y.to(device2)
        Rot_z = Rot_z.to(device2)
        Shift_x = Shift_x.to(device2)
        Shift_y = Shift_y.to(device2)
        Shift_z = Shift_z.to(device2)
        #print('The shape of the original 3D us image',us_img_3d.shape)
        #print('The shape of the 2D us image',us_img_2d.shape)
        #print('The shape of the modified 3D us image',us_img_3d_deformable.shape)

        vol_resampled, deformation_params = net(us_img_3d, us_img_2d, device = device2)
        #print('checking The grad of the resampled volume',vol_resampled.requires_grad)
        #print('checking The grad of the deformation_params',deformation_params.requires_grad)
        deformation_params = torch.transpose(deformation_params,0,1)
        
        Loss_params = torch.mean(loss1(deformation_params[3],Rot_x)+loss1(deformation_params[4],Rot_y)
                                + loss1(deformation_params[5],Rot_z)+loss1(deformation_params[0],Shift_x)
                                + loss1(deformation_params[1],Shift_y)+loss1(deformation_params[2],Shift_z))
        
        
        #smoothness_loss = 0.0
        ## add smoothness as well over the deformation field
        #print('HHHHHH',vol_resampled.shape )
        Loss_similarity = loss3(vol_resampled[0,0,5,:,:], us_img_3d_deformable[0,0,5,:,:])
        
        total_loss = Loss_similarity+ Loss_params
        total_loss = torch.tensor(total_loss,dtype=torch.float64,requires_grad=True)
        #print('The gradient of the loss is', total_loss.grad)
        #GRAD.append(total_loss.grad)
        #for param in net.parameters():
            #Grad = torch.autograd.grad(total_loss,  param)
        
        total_loss.backward(retain_graph=True)
        #print('The gradient of the loss is', total_loss.grad)
        
        
        #GRAD.append(total_loss.grad)
        #GRAD.append(Grad)

        opt.step() 

        

        t_pbar.set_description(pbar_desc('train',epoch,epochs,Loss_similarity.item(),Loss_params.item(),total_loss.grad,total_loss.item()))

        train_losses.update(parameters_loss = Loss_params.item(), similarity_loss = Loss_similarity.item(), Gradient = total_loss.grad,total_loss=total_loss.item() )
    
    avg_loss.append(total_loss.item())

def evaluate(net, val_dl,epoch,epochs,loss1,loss2,loss3,val_losses,device1,device2,device3,TENSORBOARD_LOGDIR,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,best_loss):
    
    
    v_pbar = tqdm(val_dl, desc=pbar_desc('val',epoch,epochs,0.0,0.0,0.0,0.0))
    
   
    net.eval()
    
    avg_loss = []
    
   
    with torch.no_grad():
        no = 0
        
        for us_img_3d, us_img_2d, us_img_3d_deformable,Rot_x,Rot_y,Rot_z,Shift_x,Shift_y,Shift_z,name in v_pbar:
            
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
          
            v_pbar.set_description(pbar_desc('val',epoch,epochs,Loss_similarity.item(),Loss_params.item(),smoothness_loss,total_loss.item()))


            vol_resampled =vol_resampled.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_3d = us_img_3d.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_2d = us_img_2d.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_3d_deformable = us_img_3d_deformable.detach().cpu().squeeze(0).squeeze(0).numpy()
            deformation_params = deformation_params.detach().cpu().squeeze(0).squeeze(0).numpy()
           
            if no==10 or no == 100 or no == 200:
                #n1,_= name[0].split(sep='.') 
                #name = n1
                
                vol_resampled_new = nib.Nifti1Image(vol_resampled, np.eye(4)) 
                vol_resampled_new.header.get_xyzt_units()
                vol_resampled_new.to_filename('./Result/vol_resampled_'+name[0]+'.nii.gz')
            
                us_img_3d_new  = nib.Nifti1Image(us_img_3d, np.eye(4)) 
                us_img_3d_new.header.get_xyzt_units()
                us_img_3d_new.to_filename('./Result/us_img_3d_'+name[0]+'.nii.gz')
            
                us_img_2d_new = nib.Nifti1Image(us_img_2d, np.eye(4)) 
                us_img_2d_new.header.get_xyzt_units()
                us_img_2d_new.to_filename('./Result/us_img_2d_'+name[0]+'.nii.gz')
            
                us_img_3d_deformable_new = nib.Nifti1Image(us_img_3d_deformable, np.eye(4)) 
                us_img_3d_deformable_new.header.get_xyzt_units()
                us_img_3d_deformable_new.to_filename('./Result/us_img_3d_deformable_'+name[0]+'.nii.gz')
            
                #### TO DO ####
                #### SAVE THE PREDICTED PARAMETERS ##########
                
                
            avg_loss.append(total_loss.item())
       
            
            
        avg_loss = torch.mean(torch.tensor(avg_loss))
        

        
        avg_loss = torch.mean(torch.tensor(avg_loss))
        
        val_losses.update(parameters_loss = Loss_params.item(), similarity_loss = Loss_similarity.item(),Gradient = smoothness_loss,total_loss=total_loss.item())   
        if avg_loss < best_loss or epoch % SAVE_EVERY == 0:
            best_loss = avg_loss
            torch.save(net.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-net-epoch-{epoch:04d}_{best_loss:.3f}.pth.tar')

    return best_loss


