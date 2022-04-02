import argparse

EXP_NO = 1
parser = argparse.ArgumentParser(description='Select the values of the hyper-parameters')
parser.add_argument('--TRAIN_BATCH_SIZE', default = 2)
parser.add_argument('--VAL_BATCH_SIZE', default = 1)
parser.add_argument('--LR', default = 0.0001)
parser.add_argument('--WORKERS', default = 8)
parser.add_argument('--DEVICE1', default = 'cuda:1')
parser.add_argument('--DEVICE2', default = 'cuda:0')
parser.add_argument('--DEVICE3', default = 'cuda:2')

parser.add_argument('--LR_DECAY', default = 0.5)
parser.add_argument('--LR_STEP', default = 10000)


parser.add_argument('--TRAIN_US_3D', default = '../us_data/TRAIN/orig_data/')
parser.add_argument('--TRAIN_US_3D_DEFORMED', default = '../us_data/TRAIN/generated_3D_data/')
parser.add_argument('--TRAIN_US_2D', default = '../us_data/TRAIN/generated_2D_slice/')


parser.add_argument('--VAL_US_3D', default = '../us_data/VAL/orig_data/')
parser.add_argument('--VAL_US_3D_DEFORMED', default = '../us_data/VAL/generated_3D_data/')
parser.add_argument('--VAL_US_2D', default = '../us_data/VAL/generated_2D_slice/')

parser.add_argument('--PARAMS', default = '../us_data/params.csv')
#parser.add_argument('--TEST', default = '/home/Drive3/sahar_datasets/BraTSReg_Training_Data_v2/TEST')


parser.add_argument('--EXP_NO', default = EXP_NO)
parser.add_argument('--LOAD_CHECKPOINT', default =None)#'checkpoint.pth.tar')#None)

parser.add_argument('--TENSORBOARD_LOGDIR', default = f'{EXP_NO:02d}-tboard')
parser.add_argument('--END_EPOCH_SAVE_SAMPLES_PATH', default = f'{EXP_NO:02d}-epoch_end_samples')
parser.add_argument('--WEIGHTS_SAVE_PATH', default = f'{EXP_NO:02d}-weights')
parser.add_argument('--LOSS_WEIGHT', default = 0.5)
parser.add_argument('--BATCHES_TO_SAVE', default = 1)
parser.add_argument('--SAVE_EVERY', default = 1)
parser.add_argument('--VISUALIZE_EVERY', default = 10)
parser.add_argument('--EPOCHS', default = 500)

args = parser.parse_args()

