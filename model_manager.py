import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *
import cv2 as cv
import os
import sys

def get_string(*args):
    string = ''
    for s in args:
        string = string + ' ' + str(s)
    return string

def get_prediction_name(num):
    num = str(num)
    prefix = '0' * (5 - len(num))
    suffix = '.png'
    name = prefix + num + suffix
    return name

class Manaeger():
    def __init__(self, model, args):
        
        load_name = args.load
        if load_name != None:
            weight = torch.load(load_name)
            model.load_state_dict(weight)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.lr = args.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.sigma = args.sigma
        self.save_name = '../weights/' + args.save
        self.log_file = open('logs/' + args.log, 'w')
        self.check_batch_num = args.check_batch_num
        self.pred_dir = args.predict_dir
        self.best = {'epoch':0, 'error': sys.maxsize}
    
    def load_data(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def record(self, message):
        self.log_file.write(message)
        print(message, end='')

    def get_info(self):
        info = get_string('\nModel:', self.model.name(), '\n')
        info = get_string(info, 'Learning rate:', self.lr, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, 'Weight name:', self.save_name, '\n')
        info = get_string(info, 'Log file:', self.log_file, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        info = self.get_info()
        self.record(info)
        
        for epoch in range(self.epoch_num):
            self.model.train()
            
            for batch_id, imgs in enumerate(self.train_loader):
                imgs = imgs.to(self.device)
    
                out, mu, logvar = self.model(imgs)
                loss = vae_loss(self.sigma, out, imgs, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch_id % self.check_batch_num == 0):
                    result = get_string('Epoch',epoch, '| batch', batch_id, '| Training loss :', loss.item()/(64*64),'\n')
                    self.record(result)

            self.validate(epoch)
        

    def validate(self, epoch):
        self.model.eval()
        self.record('\n=====================\n')
        loss_total = 0
        for i, imgs in enumerate(self.valid_loader):
            imgs = imgs.to(self.device)
            out, mu, logvar = self.model(imgs)
            loss = vae_loss(self.sigma, out, imgs, mu, logvar)
            loss_total += loss

        loss = loss_total / ((i+1) * 64 * 64)
        info = get_string('Validation error for', epoch, 'epoch:', loss.item())
        self.record(info)

        if loss.item() < self.best['error']:
            self.best['epoch'] = epoch
            self.best['error'] = loss.item()
            torch.save(self.model.state_dict(), self.save_name)
            self.record('\n*** Save BEST model ***\n')

        info = get_string('Best model is at epoch',self.best['epoch'], 'with error:', self.best['error'])
        self.record(info) 
        self.record('\n=====================\n')           

    def predict(self):
        self.model.eval()
        for i, imgs in enumerate(self.valid_loader):
            imgs = imgs.to(self.device)
            out = self.model(imgs)
            self.save_predictoin_for_batch(out, i)
            if i == 0: break

        sample = self.model.sample(10)
        self.save_predictoin_for_batch(sample, i, 'sample_')

    def save_predictoin_for_batch(self, batch, num, prefix = ''):
        for i,tensor in enumerate(batch):
            img = image_tensor_to_numpy(tensor)
            name = os.path.join(self.pred_dir, prefix + get_prediction_name(num + i))  
            cv.imwrite(name, img)