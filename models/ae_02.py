import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
    
        self.relu = nn.ReLU(inplace= True)

        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, padding= 1)
        self.pool1 = nn.MaxPool2d((4, 4), stride= (4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.pool2 = nn.MaxPool2d((4, 4), stride= (4, 4))
        self.conv3 = nn.Conv2d(64, 256, kernel_size= 3, padding= 1)
        self.pool3 = nn.MaxPool2d((2, 2), stride= (2, 2))
        self.conv4 = nn.Conv2d(256, 1024, kernel_size= 3, padding= 1)
        self.pool4 = nn.MaxPool2d((2, 2), stride= (2, 2))

        self.deconv1 = nn.ConvTranspose2d(1024, 256, kernel_size= 2, stride= 2)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size= 2, stride= 2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size= 4, stride= 4)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size= 4, stride= 4)

    def name(self):
        return 'AE_02 -- with skip connection'

    def forward(self, x):
        x = self.relu(self.conv1(x))
        down1 = self.pool1(x)
        x = self.relu(self.conv2(down1))
        down2 = self.pool2(x)
        x = self.relu(self.conv3(down2))
        down3 = self.pool3(x)
        x = self.relu(self.conv4(down3))
        down4 = self.pool4(x)

        out = self.relu(self.deconv1(down4))
        out += down3
        out = self.relu(self.deconv2(out))
        out += down2
        out = self.relu(self.deconv3(out))
        out += down1
        out = self.relu(self.deconv4(out))

        return out
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 8
    img_batch = torch.zeros(batch_size, 3, 64, 64)
    model = Model()
    out = model(img_batch)
    print('Parameter number: ',parameter_number(model))
    print('Input size: ', img_batch.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()