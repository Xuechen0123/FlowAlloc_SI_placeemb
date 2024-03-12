import os

import warnings

from train import main_worker
from utils import get_args_parser,set_seed

def run_train(config):
    # if train with mixed precision or not
    if config.fp16_precision:
        warnings.warn('You have choose mixed precision training. This may influence precision.')
    # multitask prefix
    if config.optimize_in and config.optimize_out:
        config.multitask_prefix='in_out'
    elif config.optimize_in:
        config.multitask_prefix='in'
    elif config.optimize_out:
        config.multitask_prefix='out'
    else:
        print('No specified task!')
        return
    
    if (config.gpu is not None):
        warnings.warn('Training on a specific GPU!')
    else:
        warnings.warn('Training on CPU!')
        
    main_worker(config)

        
def main(config):
    # some hyper-perameters
    #config.seed=99
    config.data_folder=r'/root/Experiments/simulation_flow/data'
    config.result_folder=r'/root/Experiments/simulation_flow/results'
    config.flow_hidden_dim_list=[]
    #config.distributed=False
    #config.geo_prefix='35620_New_York_Newark_Jersey_City_NY_NJ_PA'
    #config.time_prefix='2019'
    #config.optimize_out=True
    #config.optimize_in=True
    #config.gpu=2
    #config.batch_size=32
    #config.lr=2e-4
    #config.num_topk=400
    #config.num_nonzeros=150
    #config.num_zeros=50
    #config.lambda_flow=0
    # if seed training or not
    if config.seed is not None:
        set_seed(config.seed)
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting from checkpoints.')
    
    
    # train
    run_train(config)

if __name__=='__main__':
    os.environ['TORCH_DISTRIBUTED_DEBUG']='INFO'
    os.environ["CUDA_LAUNCH_BLOCKING"]='1'
    args=get_args_parser('Interaction Modeling')
    config=args.parse_args()
    print('hello')
    main(config)
    print('bye')