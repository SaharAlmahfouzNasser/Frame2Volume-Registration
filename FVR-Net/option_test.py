import argparse

parser = argparse.ArgumentParser(description='Select the values of the hyper-parameters')
parser.add_argument('--TEST_BATCH_SIZE', default = 1)
parser.add_argument('--WORKERS', default = 8)
parser.add_argument('--DEVICE1', default = 'cuda:1')
parser.add_argument('--DEVICE2', default = 'cuda:0')
parser.add_argument('--DEVICE3', default = 'cuda:2')


parser.add_argument('--TEST_US_3D', default = '../us_data/TEST/orig_data/')
parser.add_argument('--TEST_US_3D_DEFORMED', default = '../us_data/TEST/generated_3D_data/')
parser.add_argument('--TEST_US_2D', default = '../us_data/TEST/generated_2D_slice/')


parser.add_argument('--PARAMS', default = '../us_data/params.csv')

parser.add_argument('--LOAD_CHECKPOINT', default ='checkpoint.pth.tar')#None)

parser.add_argument('--EPOCHS', default = 1)

parser.add_argument('--SAVE_PATH', default = 'Result_test')

args = parser.parse_args()

