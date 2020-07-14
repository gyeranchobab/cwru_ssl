import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.enc_layer1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.enc_layer2 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer3 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer4 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        
        self.classifier1 = nn.Linear(32, 32)
        self.classifier2 = nn.Linear(32, 32)
        self.classifier3 = nn.Linear(32, 6)
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m)==nn.Linear or type(m)==nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x, get_latent=False):
        x = self.enc_layer1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(p=0.3)(x)
        
        x = self.enc_layer2(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(p=0.3)(x)
        
        x = self.enc_layer3(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(p=0.3)(x)
        
        x = self.enc_layer4(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(p=0.3)(x)
        
        x = x.mean(dim=-1)
        z=x
        
        x = self.classifier1(x)
        x = nn.ReLU()(x)
        
        x = self.classifier2(x)
        x = nn.ReLU()(x)
        
        x = self.classifier3(x)
        
        if get_latent:
            return x,z
        return x