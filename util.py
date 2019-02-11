import numpy as np
import torch
import cv2 as cv


def read_img_to_tensor(path):
    img = cv.imread(path)
    img = torch.tensor(img, dtype= torch.float)
    img = img.permute(2, 0, 1)
    return img

def test():
    img = read_img_to_tensor('00000.png')
    print(img)
    print(img.size())

if __name__ == '__main__':
    test()
    