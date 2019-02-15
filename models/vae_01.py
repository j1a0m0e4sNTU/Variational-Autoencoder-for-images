import torch
import torch.nn as nn
import numpy as np

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
        self.latent = 512
        self.training = False
        self.encoder = nn.Sequential(
            Conv2d(3, 32, 3),
            nn.MaxPool2d((4, 4), stride= (4, 4)),
            Conv2d(32, 64, 3),
            nn.MaxPool2d((4, 4), stride= (4, 4)),
            Conv2d(64, 256, 3),
            nn.MaxPool2d((2, 2), stride= (2, 2)),
            Conv2d(256, self.latent, 3),
            nn.MaxPool2d((2, 2), stride = (2, 2))
        )

        self.fc_mu = nn.Linear(self.latent, self.latent)
        self.fc_logvar = nn.Linear(self.latent, self.latent)
    
        self.decoder = nn.Sequential(
            ConvTransposes2d(self.latent, 256, 2),
            ConvTransposes2d(256, 64, 2),
            ConvTransposes2d(64, 32, 4),
            ConvTransposes2d(32, 3, 4)
        )

    def name(self):
        return 'VAE_01'
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu
        if self.training:
            z = z + eps.mul(std) 
        return z

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        mu, logvar = self.fc_mu(out), self.fc_logvar(out)
        out = self.reparameterize(mu, logvar)
        out = out.view(out.size(0), -1, 1, 1)
        out = self.decoder(out)
        return out, mu, logvar
    
    def sample(self, batch_size):
        latent = torch.rand(batch_size, self.latent).type(torch.float).cuda()
        # latent = (latent) * 2 
        # out = self.fc_mu(latent)
        out = latent.view(batch_size, self.latent, 1, 1)
        # out = out.view(batch_size, self.latent, 1, 1)
        out = self.decoder(out)
        return out
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 8
    img_batch = torch.zeros(batch_size, 3, 64, 64)
    model = Model()
    out, _, _ = model(img_batch)
    sample = model.sample(8)
    print('Parameter number: ',parameter_number(model))
    print('Input size: ', img_batch.size())
    print('Output size:', out.size())
    print('Sample size:', sample.size())
    
if __name__ == '__main__':
    unit_test()
