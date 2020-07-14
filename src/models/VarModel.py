import torch
import torch.nn as nn

class VarModel(nn.Module):
    def __init__(self):
        super(VarModel, self).__init__()
        self.enc_layer1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.enc_layer2 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer3 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.enc_layer4 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        
        self.enc_mu = nn.Linear(32, 32)
        self.enc_logvar = nn.Linear(32, 32)
        
        self.classifier1 = nn.Linear(32, 32)
        self.classifier2 = nn.Linear(32, 32)
        self.classifier3 = nn.Linear(32, 6)
        
        self.kld=torch.tensor(0)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m)==nn.Linear or type(m)==nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def _reparam(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    def _kld_gauss(self, mu, logvar):
        kld_element = -logvar + torch.exp(logvar) + mu**2 - 1
        return 0.5 * torch.mean(kld_element)
        
    def forward(self, x):
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
        
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)
        z = self._reparam(z_mu, z_logvar)
        
        self.kld = self._kld_gauss(z_mu, z_logvar)
        
        x = self.classifier1(z)
        x = nn.ReLU()(x)
        
        x = self.classifier2(x)
        x = nn.ReLU()(x)
        
        x = self.classifier3(x)
        
        return x