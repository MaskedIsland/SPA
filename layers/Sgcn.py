import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Aconv(nn.Module):
    def __init__(self):
        super(Aconv,self).__init__()

    def forward(self,x, A, shift):
        # x [b,c,n,t]
        Align_x = torch.roll(x, shift, dims=3) 
        out = torch.zeros_like(x).to(x.device)
        Align_x[...,:shift] = out[...,:shift]  
        x = torch.einsum('ncvl,vw->ncwl',(Align_x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class emb_trans(nn.Module):
    def __init__(self, device, n_dim=10):
        super(emb_trans, self).__init__()
        self.w = nn.Parameter(torch.eye(n_dim).to(device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.zeros(n_dim).to(device), requires_grad=True).to(device)
    def forward(self, nodevec1, nodevec2, n):
        nodevec1 = nodevec1.mm(self.w) + self.b.repeat(n, 1)
        nodevec2 = (nodevec2.T.mm(self.w) + self.b.repeat(n, 1)).T
        return nodevec1, nodevec2


class Sgcn(nn.Module):
    def __init__(self,c_in,c_out,dropout, kernel_size, dilation, device='cuda', e_dim=10):
        super(Sgcn,self).__init__()
        self.Aconv = Aconv()
        c_in = (1+kernel_size)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.e_trans = nn.ModuleList()

        for i in range(kernel_size-1):
            self.e_trans.append(emb_trans(device, e_dim))

    def forward(self,x, nodevec1, nodevec2):
        out = [x]
        x2 = x
        shift = 0
        n = nodevec1.size(0)
        
        adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        x1 = self.Aconv(x2,adp, shift)
        out.append(x1)
        shift = self.dilation
        x2 = x1  
        for i in range(self.kernel_size-1):
            nodevec1, nodevec2 = self.e_trans[i](nodevec1, nodevec2, n)
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)

            x1 = self.Aconv(x2, adp, shift)
            out.append(x1)
            shift = shift + self.dilation
            x2 = x1  

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

