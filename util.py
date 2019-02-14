import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt
from torch.nn import functional as F

def read_img_to_tensor(path):
    img = cv.imread(path)
    img = torch.tensor(img, dtype= torch.float)
    img = img.permute(2, 0, 1)
    return img

def image_tensor_to_numpy(tensor):
    tensor = tensor.permute(1, 2, 0).type(torch.uint8)
    img = tensor.cpu().numpy()
    return img

def vae_loss(sigma, prediction, gt, mu, logvar):
    loss_recons = F.mse_loss(prediction, gt)
    loss_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_recons + sigma * loss_KL
    return loss

def test():
    img = cv.imread('00000.png')
    cv.imshow('test', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img = read_img_to_tensor('00000.png')
    img = image_tensor_to_numpy(img)
    print(img.shape)
    cv.imshow('test', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    test()
    