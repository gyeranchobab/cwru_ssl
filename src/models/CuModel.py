import torch
import torch.nn as nn

class CuModel(nn.Module):
    def __init__(self):
        super(CuModel, self).__init__()
        self.enc_layer1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.enc_layer2 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer3 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer4 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        
        self.classifier1 = nn.Linear(4*32, 32)
        self.classifier2 = nn.Linear(32, 32)
        self.classifier3 = nn.Linear(32, 6)
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m)==nn.Linear or type(m)==nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x1 = self.enc_layer1(x)
        x1 = nn.ReLU()(x1)
        x1 = nn.Dropout(p=0.3)(x1)
        
        x2 = self.enc_layer2(x1)
        x2 = nn.ReLU()(x2)
        x2 = nn.Dropout(p=0.3)(x2)
        
        x3 = self.enc_layer3(x2)
        x3 = nn.ReLU()(x3)
        x3 = nn.Dropout(p=0.3)(x3)
        
        x4 = self.enc_layer4(x3)
        x4 = nn.ReLU()(x4)
        x4 = nn.Dropout(p=0.3)(x4)
        
        x1 = x1.mean(dim=-1)
        x2 = x2.mean(dim=-1)
        x3 = x3.mean(dim=-1)
        x4 = x4.mean(dim=-1)
        
        x = torch.cat([x1,x2,x3,x4],dim=-1)
        
        x = self.classifier1(x)
        x = nn.ReLU()(x)
        
        x = self.classifier2(x)
        x = nn.ReLU()(x)
        
        x = self.classifier3(x)
        
        return x