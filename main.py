import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_manager import Manaeger
import sys
sys.path.append('models')
from dataset import Dataset_mine

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='Train/Predict', choices=['train', 'predict'])
parser.add_argument('model', help='Model to be used')
parser.add_argument('-lr', help= 'Learning rate',type=float, default= 1e-3)
parser.add_argument('-batch_size', type= int, default= 64)
parser.add_argument('-epoch_num', type = int, default = 10)
parser.add_argument('-sigma', help= 'ratio of KL loss and reconstruction loss',type= float, default= 0.5)
parser.add_argument('-save', help='Name to be save' , default='mdoel.pkl')
parser.add_argument('-load', help='Weights to be load', default=None)
parser.add_argument('-log', help='Log file', default='log.txt')
parser.add_argument('-check_batch_num', help= 'How many batches to show result once', type= int, default=200)
parser.add_argument('-predict_dir', help= 'Directory which stores predicted images', default='../prediction')

args = parser.parse_args()

# Prepare datasets, data loader
dataset_trian = Dataset_mine('../face_data', train= True)
dataset_valid = Dataset_mine('../face_data', train= False)
trian_loader = DataLoader(dataset_trian, batch_size= args.batch_size, shuffle= True)
valid_loader = DataLoader(dataset_valid, batch_size= args.batch_size, shuffle= False)

def get_model(name):
    model_file = __import__(name)
    model = model_file.Model()
    return model

def main():
    print('main function is running ...')
    model = get_model(args.model)
    manager = Manaeger(model, args)
    manager.load_data(trian_loader, valid_loader)
    if args.mode == 'train':
        manager.train()
    elif args.mode == 'predict':
        manager.predict()

if __name__ == '__main__':
    main()
