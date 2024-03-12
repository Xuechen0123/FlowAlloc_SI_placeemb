import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from train_utils import *
from dataset import FlowDataset
from flowalloc_placeemb_model import FlowAllocPlaceEmbModel

def cal_loss(outflow_trans_input,inflow_trans_input,
             outflow_trans_target,inflow_trans_target,
             criterion_od,device='cpu'):
    # outflow related
    if (outflow_trans_input is not None):
        # para:(input,target)
        outflow_trans_loss=criterion_od(outflow_trans_input.view_as(outflow_trans_target),
                                        outflow_trans_target)
    else:
        outflow_trans_loss=torch.tensor([0],dtype=torch.float32,device=device)
    # inflow related
    if (inflow_trans_input is not None):
        inflow_trans_loss=criterion_od(inflow_trans_input.view_as(inflow_trans_target),
                                       inflow_trans_target)
    else:
        inflow_trans_loss=torch.tensor([0],dtype=torch.float32,device=device)
    return outflow_trans_loss,inflow_trans_loss
           
def train_epoch(placeemb_model,
                flow_dataloader,
                optimizer,
                scaler,
                criterion_od,
                epoch,
                gpu,
                config):
    # preparation for recording
    ## batch train time
    batch_time = AverageMeter('Time', ':6.3f') 
    ## total loss
    od_losses = AverageMeter('Loss', ':.4e')
    ## od loss
    out_od_losses = AverageMeter('Out_OD_Loss', ':.4e')
    in_od_losses = AverageMeter('In_OD_Loss', ':.4e')
    progress = ProgressMeter(len(flow_dataloader),
                             [batch_time, od_losses],
                             prefix="Epoch: [{}]".format(epoch))
    start=time.time()
    # train mode
    placeemb_model.train()
    # specify device info
    if gpu is None:
        device='cpu'
    else:
        device='cuda:{}'.format(gpu)
        
    # train over batch
    for i,(place_ids,\
           out_trans_target,in_trans_target) in enumerate(flow_dataloader):
        
        with torch.cuda.amp.autocast(enabled=config.fp16_precision):
            ## place_ids:(batch_size,)
            ## out_od_prob:(batch_size,num_places)
            ## outflow_tensor:(batch_size,flow_len)
            ## in_od_prob:(batch_size,num_places)
            ## outflow_tensor:(batch_size,flow_len)
            place_ids=place_ids.long().to(device)
            out_trans_target=out_trans_target.to(torch.float32).to(device)
            in_trans_target=in_trans_target.to(torch.float32).to(device)
            # forward
            out_trans_predict,in_trans_predict=placeemb_model(place_ids,
                                                              device=device)
            
            if out_trans_predict is not None:
                out_trans_predict=torch.log(torch.clamp(out_trans_predict,
                                                        min=1e-10))
            if in_trans_predict is not None:
                in_trans_predict=torch.log(torch.clamp(in_trans_predict,
                                                       min=1e-10))
            # calculate loss
            outflow_trans_loss,inflow_trans_loss=cal_loss(out_trans_predict,in_trans_predict,
                                                          out_trans_target,in_trans_target,
                                                          criterion_od,device=device)
            losses=inflow_trans_loss+outflow_trans_loss
            # od related
            od_losses.update(losses.item(),place_ids.size(0))
            out_od_losses.update(outflow_trans_loss.item(),place_ids.size(0))
            in_od_losses.update(inflow_trans_loss.item(),place_ids.size(0))
                      
        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        # update batch training info
        batch_time.update(time.time() - start)
        if (i % config.print_freq == 0):
            progress.display(i)

    return out_od_losses.avg,in_od_losses.avg,od_losses.avg
          
def main_worker(config):    
    # 1.load data 
    ## 1.1.load flow data
    flow_edge_id=[]
    flow_edge_weight=[]
    with open(config.flow_data_file,'r') as f:
        for line in f:
            line=line.strip()
            flow_edge_id.append([int(line.split('\t')[0]),
                                 int(line.split('\t')[1])])
            flow_edge_weight.append([float(line.split('\t')[2])])
    flow_edge_id=np.array(flow_edge_id)
    flow_edge_weight=np.array(flow_edge_weight)         
    flow_edge_df=pd.DataFrame(data=flow_edge_id,
                              columns=['o_place','d_place'])
    unique_o_places=flow_edge_df['o_place'].unique().tolist()
    unique_d_places=flow_edge_df['d_place'].unique().tolist()
    ## 1.2.find places with inflow or outflow  (the dict of all embeddings is built from the all the places with inflow or outflow)
    all_place_ids=list(set(unique_o_places).union(set(unique_d_places)))   
    all_place_ids.sort()
    all_place_id_series=pd.Series(index=all_place_ids,
                                  data=np.arange(len(all_place_ids)))    
    print('{} places with inflow or outflow'.format(len(all_place_ids)))                                                            
    ## 1.3.find common places, i.e., those with both inflow and outflow   
    ### for inflow and outflow allocation/ inflow allocation/ outflow allocation we all use places with both inflow and outflow                                                                     
    common_od_places=list(set(unique_o_places).intersection(unique_d_places))
    #### Iterately delete places. as in Algorithm 1
    while True:
        #### stop when num_unique_origins=num_unique_destinations=num_unique_intersectedOD
        if (len(unique_o_places)==len(common_od_places)) and (len(unique_d_places)==len(common_od_places)):
            break
        flow_edge_df=flow_edge_df[flow_edge_df['o_place'].isin(common_od_places) & \
                                  flow_edge_df['d_place'].isin(common_od_places)]
        unique_o_places=flow_edge_df['o_place'].unique().tolist()
        unique_d_places=flow_edge_df['d_place'].unique().tolist()
        common_od_places=list(set(unique_o_places).intersection(unique_d_places))
    print('{} places with inflow and outflow'.format(len(common_od_places))) 
    # 2.formalize dataset and corresponding dataloader
    ## 2.1.dataset
    flow_dataset=FlowDataset(all_place_id_series,
                             common_od_places,
                             flow_edge_id,
                             flow_edge_weight)
    print('{} places in train set'.format(len(flow_dataset)))
    ## 2.2.dataloader
    ### sampler
    flow_dataloader=get_dataloader(flow_dataset,
                                   None,config.workers,
                                   config.batch_size,
                                   flag_droplast=True)
    # 3.init model
    flowalloc_placeemb_model=FlowAllocPlaceEmbModel(config)
    ## set gpu info
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        flowalloc_placeemb_model.cuda(config.gpu)
    ## define scaler,optimizer and criterion
    optimizer = torch.optim.Adam(flowalloc_placeemb_model.parameters(),lr=config.lr,
                                 betas=(config.adam_beta1,config.adam_beta2),
                                 eps=config.epsilon,weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16_precision)
    ## criterion
    criterion_od=nn.KLDivLoss(reduction="batchmean").cuda(config.gpu)
    
    # 4.train by epoch
    for epoch in tqdm(range(0,config.num_epochs)):
        # adjust learning rate
        adjust_learning_rate(optimizer,epoch,config)
        
        # train
        train_out_od_loss,\
        train_in_od_loss,\
        train_loss=train_epoch(flowalloc_placeemb_model,
                               flow_dataloader,
                               optimizer,
                               scaler,
                               criterion_od,
                               epoch,
                               config.gpu,
                               config)

        ## save checkpoint
        ## adjusted by print_freq
        if (config.model_save_folder is not None) and ((epoch+1) % config.print_freq==0):
            save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': config.arch,
                            'state_dict': flowalloc_placeemb_model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            }, filename=os.path.join(config.model_save_folder,\
                                                    'checkpoint_{:05d}.pth.tar'.format(epoch)))
    
    # 5.save embeddings if asked
    if config.representation_save_file is not None:
        o_embeddings,d_embeddings=flowalloc_placeemb_model.get_input_embeddings()
        idx_valid_place_list=all_place_id_series.loc[common_od_places].values.tolist()
        o_embedding_arr=o_embeddings.weight.detach().cpu().numpy()[idx_valid_place_list,:]
        d_embedding_arr=d_embeddings.weight.detach().cpu().numpy()[idx_valid_place_list,:]    
        embedding_dict={'place_id':['place_{:04d}'.format(place_id) for place_id in common_od_places],
                        'origin_representation':o_embedding_arr,
                        'destination_representation':d_embedding_arr}
        with open(config.representation_save_file,'wb') as f:
            pickle.dump(embedding_dict,f)