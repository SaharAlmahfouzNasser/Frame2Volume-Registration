import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import dataloader
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image
import random
from monai.losses import LocalNormalizedCrossCorrelationLoss,DiceFocalLoss, BendingEnergyLoss,GlobalMutualInformationLoss
from monai.networks.layers import Norm
from option import args
from train import *
#from losses import *
from networks.fvrnet import mynet3
from torch.nn import init
#from torch.cuda.amp import autocast, GradScaler


TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
VAL_BATCH_SIZE = args.VAL_BATCH_SIZE
LR = args.LR
WORKERS = args.WORKERS
device1 = args.DEVICE1
device2 = args.DEVICE2
device3 = args.DEVICE3
LR_DECAY = args.LR_DECAY
LR_STEP= args.LR_STEP
TRAIN_US_3D = args.TRAIN_US_3D
TRAIN_US_3D_DEFORMED = args.TRAIN_US_3D_DEFORMED
TRAIN_US_2D = args.TRAIN_US_2D

VAL_US_3D = args.VAL_US_3D
VAL_US_3D_DEFORMED = args.VAL_US_3D_DEFORMED
VAL_US_2D = args.VAL_US_2D
PARAMS = args.PARAMS
EXP_NO = args.EXP_NO 
LOAD_CHECKPOINT = args.LOAD_CHECKPOINT

TENSORBOARD_LOGDIR = args.TENSORBOARD_LOGDIR 
END_EPOCH_SAVE_SAMPLES_PATH = args.END_EPOCH_SAVE_SAMPLES_PATH
WEIGHTS_SAVE_PATH = args.WEIGHTS_SAVE_PATH 
LOSS_WEIGHT = args.LOSS_WEIGHT
BATCHES_TO_SAVE = args.BATCHES_TO_SAVE 
SAVE_EVERY = args.SAVE_EVERY 
VISUALIZE_EVERY = args.VISUALIZE_EVERY 
EPOCHS = args.EPOCHS



def init_weights(net, init_type = 'kaiming', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            ### The find() method returns -1 if the value is not found
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
                ### notice: weight is a parameter object but weight.data is a tensor
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func) ### it is called for m iterating over every submodule of (in this case) net as well as net itself, due to the method call net.apply(…).



class Bookkeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['parameters_loss','similarity_loss','Gradient','total_loss']
        self.genesis()
        ## initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))
            
    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key]+=kwargs[key]
        self.count +=1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / (self.count +1e-10)
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)


def save_checkpoint(epoch, net, best_metrics, optimizer, lr_scheduler, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'net_state_dict': net.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)


def main():
    print("Welcome to MONAI")
    transforms = T.Compose([T.ToTensor()])
    trn_ds = dataloader.US_Dataset(TRAIN_US_3D,TRAIN_US_2D,TRAIN_US_3D_DEFORMED,PARAMS,transforms)
    trn_dl = DataLoader(trn_ds, TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    transforms = T.Compose([T.ToTensor()])#,T.CenterCrop(256)])
    val_ds = dataloader.US_Dataset(VAL_US_3D,VAL_US_2D,VAL_US_3D_DEFORMED,PARAMS,transforms)
    val_dl = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle = False, num_workers = WORKERS)
    start_epoch = 1
    layers = [3, 8, 36, 3]#[3, 4, 6, 3] 
    net = mynet3(layers)
    init_weights(net)
    
   
    print('Net parameters:', sum(p.numel() for p in net.parameters()))

    loss1 = torch.nn.L1Loss().to(device2)#torch.nn.MSELoss().to(device2)
    loss2 = BendingEnergyLoss().to(device2)
    loss3 = torch.nn.MSELoss().to(device2)# LocalNormalizedCrossCorrelationLoss(spatial_dims=2).to(device2)#GlobalMutualInformationLoss().to(device2)# DiceFocalLoss ().to(device3)#torch.nn.LogSoftmax().to(device3)
    opt = torch.optim.Adam(net.parameters(), LR,weight_decay=0.0001)
    
    sched = optim.lr_scheduler.StepLR(opt, LR_STEP, gamma=LR_DECAY)
    
    

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module = dill,map_location=device2)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net_state_dict'],strict=False)
        opt = checkpoint['optimizer']
        sched = checkpoint['lr_scheduler']
    
  
    net.to(device2)
    
    train_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='val')

    best_loss = float('inf')
    
    
    #### for mixed precision ####
    #scaler = GradScaler()
    for epoch in range(start_epoch, EPOCHS+1):
        ## training loop
        
        train(net, trn_dl,epoch,EPOCHS,loss1,loss2,loss3,opt,train_losses,device1,device2,device3,TENSORBOARD_LOGDIR,LOSS_WEIGHT,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO)

        best_loss= evaluate(net, val_dl,epoch,EPOCHS,loss1,loss2,loss3,val_losses,device1,device2,device3,TENSORBOARD_LOGDIR,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,best_loss)
        
        sched.step()
        
        save_checkpoint(epoch, net, None, opt, sched)
        

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        ## Reset all losses for the new epoch
        train_losses.reset()
        val_losses.reset()
if __name__=='__main__':
    main()
