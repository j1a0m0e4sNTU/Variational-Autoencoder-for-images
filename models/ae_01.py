import torch
import torch.nn as nn

def Conv2d(in_planes, out_planes, kernel_size = 3):
    layer = nn.Sequential(
        nn.Conv2d(in_channels= in_planes, out_channels= out_planes, kernel_size= kernel_size, padding= (kernel_size-1)//2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace= True)
    )
    return layer

def ConvTransposes2d(in_planes, out_planes, kernel_size):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels= in_planes, out_channels= out_planes, kernel_size= kernel_size, stride= kernel_size),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace= True)
    )
    return layer

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder = nn.Sequential(
            Conv2d(3, 32, 3),
            nn.MaxPool2d((4, 4), stride= (4, 4)),
            Conv2d(32, 64, 3),
            nn.MaxPool2d((4, 4), stride= (4, 4)),
            Conv2d(64, 256, 3),
            nn.MaxPool2d((2, 2), stride= (2, 2)),
            Conv2d(256, 1024, 3),
            nn.MaxPool2d((2, 2), stride = (2, 2))
        )
    
        self.decoder = nn.Sequential(
            ConvTransposes2d(1024, 256, 2),
            ConvTransposes2d(256, 64, 2),
            ConvTransposes2d(64, 32, 4),
            ConvTransposes2d(32, 3, 4)
        )

    def name(self):
        return 'AE_01'

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
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