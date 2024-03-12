import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self,all_place_id_series,
                      common_od_places,
                      flow_edge_id,
                      flow_edge_weight):
        super().__init__()
        
        assert flow_edge_id.shape[0]==flow_edge_weight.shape[0]
        assert flow_edge_id.shape[1]==2
        assert flow_edge_weight.shape[1]==1
        
        self.all_place_id_series=all_place_id_series
        self.common_od_places=common_od_places
        self.num_places=len(self.all_place_id_series)
        ## flow_edge_id:(num_edges,2)
        ## flow_edge_weight:(num_edges,1)
        self.flow_edge_id=flow_edge_id
        self.flow_edge_weight=flow_edge_weight
        
    def __len__(self,):
        return len(self.common_od_places)
    
    def __getitem__(self,index):
        in_trans_flow_arr=np.zeros(self.num_places)
        out_trans_flow_arr=np.zeros(self.num_places)
        place_idx=self.common_od_places[index]
        # 1.deal with all outflows from place_idx
        with_outflow_edge=np.where(self.flow_edge_id[:,0]==place_idx)
        
        out_flow_edge_arr=self.flow_edge_id[with_outflow_edge]
        out_flow_weight_arr=self.flow_edge_weight[with_outflow_edge]
        ## check
        assert np.all(out_flow_edge_arr[:,0]==place_idx)
        out_flow_dest_place_id=out_flow_edge_arr[:,1]
        out_trans_flow_arr[self.all_place_id_series\
                               .loc[out_flow_dest_place_id].tolist()]=out_flow_weight_arr.squeeze()
        ## calculate total outflow and out_trans_prob
        total_outflow=np.sum(out_trans_flow_arr)
        out_trans_prob=out_trans_flow_arr/np.sum(out_trans_flow_arr)
        
        # 2.deal with all inflows from place_idx
        with_inflow_edge=np.where(self.flow_edge_id[:,1]==place_idx)
        
        in_flow_edge_arr=self.flow_edge_id[with_inflow_edge]
        in_flow_weight_arr=self.flow_edge_weight[with_inflow_edge]
        ## check
        assert np.all(in_flow_edge_arr[:,1]==place_idx)
        in_flow_origin_place_id=in_flow_edge_arr[:,0]
        in_trans_flow_arr[self.all_place_id_series.\
                               loc[in_flow_origin_place_id].tolist()]=in_flow_weight_arr.squeeze()
        ## calculate total inflow
        total_inflow=np.sum(in_trans_flow_arr)
        in_trans_prob=in_trans_flow_arr/np.sum(in_trans_flow_arr)
        
        return self.all_place_id_series.loc[place_idx],\
               out_trans_prob,in_trans_prob
        