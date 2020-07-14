import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b

class ENC(nn.Module):
    def __init__(self, in_dim, h_dim, n_mode, enc_depth, n_label=6):
        super(ENC, self).__init__()
        self.depth = enc_depth
        self.in_dim=in_dim
        self.encoders = nn.ModuleList([])
        self.encoders.append(nn.Conv1d(in_dim, h_dim, kernel_size=8, stride=4))
        self.decoders = nn.ModuleList([])
        self.decoders.append(nn.ConvTranspose1d(2*h_dim, in_dim, kernel_size=8, stride=4))
        for i in range(enc_depth-1):
            self.encoders.append(nn.Conv1d(h_dim, h_dim, kernel_size=8, stride=4))
            self.decoders.append(nn.ConvTranspose1d(2*h_dim,h_dim,kernel_size=8, stride=4))
        self.multi_enc=_build_mmlp(h_dim*enc_depth, h_dim, h_dim, n_mode)
        self.multi_dec=_build_mmlp(h_dim,h_dim,h_dim,n_mode)
        self.calc_alpha = MultimodalLinear(h_dim, 1, n_mode)
        self.enc_mu = nn.Linear(h_dim, h_dim)
        self.n_mode=n_mode
        classify=[]
        for i in range(2):
            classify.append(nn.Linear(h_dim,h_dim))
            classify.append(nn.LeakyReLU())
        classify.append(nn.Linear(h_dim,n_label))
        self.classify = nn.Sequential(*classify)
        self.apply(self.init_weights)
    def init_weights(self,m):
        if type(m) == nn.Linear or type(m)==nn.Conv1d or type(m)==nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def enc(self, x):
        #x shape : (batch,3,8192)
        n_sensor = x.size(1)
        x = x.reshape(x.size(0)*x.size(1),1,x.size(-1))
        #x shape : (batch*3, 1, 8192)
        f_l=[]
        x_l=[]
        self.sizes=[]
        for i in range(self.depth):
            #x shape : (batch*3, dim, len)
            self.sizes.append(x.size())
            x = self.encoders[i](x)
            x = nn.LeakyReLU().cuda()(x)
            x = nn.Dropout(p=0.5)(x)
            f_l.append(x)
            x_l.append(torch.mean(x.view(-1,n_sensor,x.size(1),x.size(2)),dim=-1))
        self.comp_len = x.size(2)
        x = torch.cat(x_l,dim=-1)
        return x, f_l
    def dec(self, x, f_l):
        #x shape: (batch, 3, dim)
        x = x.view(x.size(0)*x.size(1),x.size(2),1)
        x = x.repeat(1,1,self.comp_len)
        for i in range(self.depth):
            x = torch.cat([x,f_l[-i-1]],dim=1)
            x = self.decoders[-i-1](x, output_size=self.sizes[-i-1])
            x = nn.ReLU().cuda()(x)
        x = x.view(-1,self.n_mode, x.size(-1))
        return x
    def encode(self, x, m):
        #x shape : (batch, 3, 8192)
        f, f_l = self.enc(x)
        #f shape : (batch, 3, dim)
        f = self.multi_enc(f)
        alpha = self.calc_alpha(f)
        alpha[~m] = float('-inf')
        alpha = nn.Softmax(dim=1)(alpha)
        #alpha shape : (batch, 3,1)
        f = (alpha*f).sum(dim=1)
        #f shape : (batch, dim)
        mu = self.enc_mu(f)
        #logvar = self.enc_logvar(f)
        #mu/logvar shape : (batch, dim)
        z = mu#self.reparameterization(mu, logvar)
        self.f_l = f_l
        return z
    def decode(self, z, f_l):
        #z shape : (batch, dim)
        z = z.unsqueeze(1).repeat(1,self.n_mode,1)
        f = self.multi_dec(z)
        #f shape : (batch, 3, dim)
        return self.dec(f, f_l)
    
        
    def forward(self,x,m,label,label_mask):
        z = self.encode(x,m)
        #z shape : (batch, dim)
        y_recon = self.decode(z, self.f_l)
        y_class = self.classify(z)
        pred = torch.argmax(y_class,dim=1)
        
        recon_loss = nn.MSELoss()(y_recon, x)
        class_loss = torch.tensor(0.0).to(device=y_class.device)
        if label_mask.any():
            class_loss += nn.CrossEntropyLoss()(y_class[label_mask], label[label_mask])
        entropy_loss = HLoss()(y_class)
        return recon_loss, class_loss, entropy_loss, pred
    def predict(self, x, m):
        z = self.encode(x,m)
        y = self.classify(z)
        return torch.argmax(y,dim=1)
def _build_mmlp(in_dim, out_dim, h_dim, n_mode,n_layers=3, bn=True):
    layers=[]
    layers.append(MultimodalLinear(in_dim, h_dim,n_mode))
    layers.append(nn.LeakyReLU())
    for l in range(n_layers-2):
        layers.append(MultimodalLinear(h_dim,h_dim,n_mode))
        layers.append(nn.LeakyReLU())
    layers.append(MultimodalLinear(h_dim,out_dim,n_mode))
    return nn.Sequential(*layers)

class MultimodalLinear(nn.Module):
    def __init__(self, in_features, out_features, n_mode, bias=True):
        super(MultimodalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(n_mode, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(n_mode, out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, input):
        output = self.weight*input.unsqueeze(-1)
        output = output.sum(dim=-2)
        if self.bias is not None:
            output += self.bias
        return output