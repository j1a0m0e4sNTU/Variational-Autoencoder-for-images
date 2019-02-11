import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
    
    def name(self):
        return 'base_01'

    def forward(self, x):
        
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 8
    img_batch = torch.zeros(batch_size, 1, 28, 28)
    model = Model()
    out = model(img_batch)
    print('Parameter number: ',parameter_number(model))
    print('Input size: ', img_batch.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()