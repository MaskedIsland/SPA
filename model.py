import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.TTransformer import TTransformer
from layers.Sgcn import Sgcn, emb_trans

class SpaBlock(nn.Module):
    def __init__(self, embed_size, heads, time_num, t_dropout, forward_expansion, dilation=1, kernel_size=2, a_dropout = 0.3, e_dim= 10):
        super(SpaBlock, self).__init__()
        
        self.TTransformer = TTransformer(embed_size, heads, time_num, t_dropout, forward_expansion)
        self.Sgcn = Sgcn(embed_size, embed_size, dropout=a_dropout, kernel_size = kernel_size, dilation = dilation, e_dim = e_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(t_dropout)
    
    def forward(self, query, t, nodevec1, nodevec2):
        # value,  key, query: [N, T, C] [B, N, T, C]
        x = query.permute(0,3,1,2) # [B, C, N, T]
        x1 = self.norm1(self.Sgcn(x, nodevec1, nodevec2).permute(0,2,3,1) + query) #(B, N, T, C)
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t) + x1) ) 
        return x2
 

### SPA: Total Model

class SPA(nn.Module):
    def __init__(
        self, 
        num_nodes=307,  
        in_channels = 1,  
        embed_size = 64,   
        time_num = 288,    
        num_layers = 4,  
        T_dim = 12,   
        output_T_dim = 12,  
        heads = 2,    
        forward_expansion=4,   
        t_dropout = 0,  
        kernel_size=3,  
        dilation=1,   
        a_dropout=0.3, 
        e_dim = 10, 
        device='cuda'
    ):        
        super(SPA, self).__init__()
        self.forward_expansion = forward_expansion
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)

        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                SpaBlock(
                    embed_size,
                    heads,
                    time_num,
                    t_dropout=t_dropout,
                    forward_expansion=forward_expansion,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    a_dropout = a_dropout,
                    e_dim = e_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.e_trans = nn.ModuleList(
            [
                emb_trans(device, e_dim) for _ in range(num_layers-1)
            ]
        )

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, e_dim).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(e_dim, num_nodes).to(device), requires_grad=True).to(device)
        self.dropout = nn.Dropout(t_dropout)


        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
 
        input_Transformer = self.conv1(x).permute(0,2,3,1)          
        #input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.dropout(input_Transformer)        
        n1 = self.nodevec1
        n2 = self.nodevec2
        n = self.nodevec1.size(0)
        output_Transformer = self.layers[0](output_Transformer, self.forward_expansion, n1, n2)
        for i in range(len(self.layers)-1):
            n1, n2 = self.e_trans[i](n1, n2, n)
            output_Transformer = self.layers[i+1](output_Transformer, self.forward_expansion, n1, n2)
        


        # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        out = self.relu(self.conv2(output_Transformer))    # [B, output_T_dim, N, C]        
        out = out.permute(0, 3, 2, 1)           # [B, C, N, output_T_dim]
        out = self.conv3(out)                   # [B, 1, N, output_T_dim]   
        # out = out.squeeze(1)
        out = out.permute(0,3,2,1)
        return out # B, output_T_dim, N,
