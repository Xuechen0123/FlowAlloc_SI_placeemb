import os
import torch
import random
import argparse
import numpy as np

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    
def get_args_parser(descrip_str='Interaction Model Training'):
    parser = argparse.ArgumentParser(description=descrip_str)
    
    # data info
    ## place/flow data
    parser.add_argument('--place-data-file',type=str,
                        default='example_flow/gravity_place_data.csv',
                        help='path to the place data')
    parser.add_argument('--flow-data-file',type=str,
                        default='example_flow/gravity_flow_data.txt',
                        help='path to the flow data')
    ## result folder
    parser.add_argument('--model-save-folder',type=str,default=None,
                        help='path to save training result')
    parser.add_argument('--representation-save-file',type=str,default=None,
                        help='path to save representation result')
    ## simulate place info
    parser.add_argument('--num-places',type=int,default=None,
                        help='number of places in simulated data')
    
    # model hyper-parameters related
    parser.add_argument('--dropout',type=float,default=0.1,help='Dropout rate')
    parser.add_argument('--place-embed-dim',type=int,default=32,
                        help='Embedding dim of places')
    
    # training process related hyper-parameters
    ## distributed training related hyper-parameters
    parser.add_argument('--seed',default=None,type=int,help='seed for training')
    parser.add_argument('--gpu',default=None,type=int,help='assign a GPU for training')
    parser.add_argument('--workers',type=int,default=0,help='number of workers to use')

    ## training related hyper-parameters
    ### different learning rates for different groups of parameters
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate')          
    parser.add_argument('--adam-beta1',type=float,default=0.9,help='Beta1 parameter of Adam optimizer')
    parser.add_argument('--adam-beta2',type=float,default=0.999,help='Beta2 parameter of Adam optimizer')
    parser.add_argument('--epsilon',type=float,default=1e-8,help='Epsilon parameter of Adam optimizer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,metavar='W', 
                        help='weight decay (default: 1e-4)',dest='weight_decay')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_false',
                        help='use cosine lr schedule')
    parser.add_argument('--batch-size',default=32,type=int,
                         help='batch size of  num places for training')
    parser.add_argument('--num-epochs',type=int,default=500,
                        help='Number of epochs to train the model')
    parser.add_argument('--warmup-epochs',type=int,default=10,
                        help='Number of epochs to train the model')
    parser.add_argument('--print-freq',type=int,default=10,help='frequency of result print')
    parser.add_argument('--fp16-precision',action='store_true',
                        dest='fp16_precision',help='train with mixed precision or not')
    parser.add_argument('--optimize-out',action='store_true',
                        dest='optimize_out',help='optimize outflow or not')
    parser.add_argument('--optimize-in',action='store_true',
                        dest='optimize_in',help='optimize inflow or not')


    return parser
