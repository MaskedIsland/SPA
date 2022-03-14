import os
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser

from datetime import datetime

from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch

from utils import normalized_network, get_adjacency_matrix
from train import Trainer
from model import SPA

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=True, type=eval)
    parser.add_argument('--dataset', default='PeMSD4', type=str)
    parser.add_argument('--model', default='SPA', type=str)
    #train
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--loss_func', default='mae', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--epochs',type=int,default=200,help='')
    parser.add_argument('--lr_init', default=0.001, type=float)
    parser.add_argument('--lr_decay', default=False, type=eval)
    parser.add_argument('--lr_decay_rate', default=0.3, type=float)
    parser.add_argument('--lr_decay_step', default='5,20,40,70', type=str)
    parser.add_argument('--early_stop', default=True, type=eval)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--grad_norm', default=False, type=eval)
    parser.add_argument('--max_grad_norm', default=5, type=int)
    parser.add_argument('--teacher_forcing', default=False, type=bool)
    parser.add_argument('--real_value', default=True, type=eval, help = 'use real value for loss calculation')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')

    #test
    parser.add_argument('--mae_thresh', default=None, type=eval)
    parser.add_argument('--mape_thresh', default=0., type=float)
    #log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--plot', default=False, type=eval)

    # model
    parser.add_argument('--in_channels',type=int,default=1,help='inputs dimension')
    parser.add_argument('--embed_size',type=int,default=64,help='')
    parser.add_argument('--time_num',type=int,default=288,help='')
    parser.add_argument('--heads',type=int,default=2,help='')
    parser.add_argument('--num_layers',type=int,default=4,help='')
    parser.add_argument('--forward_expansion',type=int,default=4,help='')
    parser.add_argument('--t_dropout',type=int,default=0,help='')
    parser.add_argument('--kernel_size',type=int,default=3,help='')
    parser.add_argument('--a_dropout',type=int,default=0.3,help='')

    # data
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--num_nodes', default=307, type=int)
    parser.add_argument('--tod', default=False, type=eval)
    parser.add_argument('--normalizer', default='std', type=str)
    parser.add_argument('--column_wise', default=False, type=eval)
    parser.add_argument('--default_graph', default=False, type=eval)

    parser.add_argument('--input_dim', default=1, type=int)  
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--e_dim', default=10, type=int)

    args = parser.parse_args()

    init_seed(args.seed)

    device = torch.device(args.device)

    train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                      normalizer=args.normalizer,
                                      tod=args.tod, dow=False,
                                      weather=False, single=False)

    if args.default_graph:
        adjinit, distance = get_adjacency_matrix(('./data/{}/distance.csv').format(args.dataset), args.num_nodes)
    else:
        adjinit = None 
    
    if args.model == 'SPA':
        model = SPA(num_nodes=args.num_nodes, in_channels = args.in_channels, embed_size = args.embed_size, time_num = args.time_num, num_layers = args.num_layers, 
                T_dim = args.lag, output_T_dim = args.horizon, heads = args.heads, forward_expansion = args.forward_expansion, 
                t_dropout = args.t_dropout, kernel_size = args.kernel_size, a_dropout = args.a_dropout, e_dim=args.e_dim)
    
    model = model.to(device)
    print_model_parameters(model, only_num=False)
    #init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                weight_decay=args.weight_decay)

    #learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)

    
    #config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.join('./log')
    log_dir = os.path.join(current_dir,'experiments', args.dataset, '{}_{}_{}_{}_{}_{}_{}'.format(
            current_time, args.model, args.lr_init, args.input_dim, args.num_nodes, args.kernel_size, args.num_layers))
    args.log_dir = log_dir

    #start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                    args, lr_scheduler=lr_scheduler)
    trainer.train()
    