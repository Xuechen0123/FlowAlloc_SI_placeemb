import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset,DataLoader


def get_sampler(dataset:Dataset,config):
    if dataset is None:
        return None
    if config.distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        return None

def get_dataloader(dataset:Dataset,
                   sampler,num_workers,batch_size,
                   flag_droplast=True):
    if dataset is None:
        return None
    return DataLoader(dataset,batch_size=batch_size,
                      shuffle=(sampler is None),sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=False,drop_last=flag_droplast)

def adjust_learning_rate(optimizer, epoch, config,
                         lr=None,
                         warmup_epochs=None):
    """Decay the learning rate based on schedule"""
    if lr is None:
        lr = config.lr
    if warmup_epochs is None:
        warmup_epochs=config.warmup_epochs
    
    if epoch < warmup_epochs:
        lr*=(epoch+1)/warmup_epochs
    else:
        if config.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch-warmup_epochs)\
                                    / (config.num_epochs-warmup_epochs)))
        else:  # stepwise lr schedule
            for milestone in config.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state,filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def modify_statedict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name=k.replace('module.','')# remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch,extra_info=None):
        #entries = ['parent level: {}'.format(parent_level)]
        entries  = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if extra_info is not None:
            entries+=[extra_info]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_checkpoint(checkpoint_path,model,gpu=None):
    if gpu is None:
        loc='cpu'
    else:
        loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    model.load_state_dict(modify_statedict(checkpoint['state_dict']))
    return model