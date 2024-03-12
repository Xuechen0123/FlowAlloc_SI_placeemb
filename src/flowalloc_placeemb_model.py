import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceEmbeddings(nn.Module):
    def __init__(self,num_places,
                      place_embed_dim,
                      dropout=0.1):
        super().__init__()
        
        self.num_places=num_places
        self.place_embed_dim=place_embed_dim
        self.layernorm=nn.LayerNorm(place_embed_dim)
        self.dropout=nn.Dropout(dropout)

        self.place_embeddings=nn.Embedding(num_places,place_embed_dim)

    def forward(self,place_ids):
        x=self.layernorm(self.place_embeddings(place_ids))
        return self.dropout(x)
    
def attn_places(query_places,context_places):
    # calculate place_attn_result first
    ## query_places:(batch_size,1,place_embed_dim)
    ## context_places:(batch_size,num_places,place_embed_dim)->(batch_size,place_embed_dim,num_places)
    place_attn_result=torch.bmm(query_places,
                                torch.transpose(context_places,1,2))
    # normalize to get attn_weight
    od_attn_weights=F.softmax(place_attn_result,dim=2)
    # add places by attn_weight
    ## od_attn_weights:(batch_size,1,num_places)
    ## context_places:(batch_size,num_places,place_embed_dim)
    ## place_output:(batch_size,1,place_embed_dim)->(batch_size,place_embed_dim)
    place_output=torch.bmm(od_attn_weights,context_places).squeeze(1)
    return place_output,od_attn_weights.squeeze(1)

class FlowAllocPlaceEmbModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config=config
        
        self.num_places=config.num_places
        self.place_embed_dim=config.place_embed_dim
        self.optimize_out=config.optimize_out
        self.optimize_in=config.optimize_in
        self.dropout=config.dropout

        # target place embeddings
        self.o_embeddings=PlaceEmbeddings(config.num_places,
                                          config.place_embed_dim)
        self.d_embeddings=PlaceEmbeddings(config.num_places,
                                          config.place_embed_dim)
        # init parameters
        self.init_paras()
        
    def set_input_embeddings(self,value_o,value_d):
        self.o_embeddings.place_embeddings=value_o
        self.d_embeddings.place_embeddings=value_d

    def get_input_embeddings(self,):
        return self.o_embeddings.place_embeddings,\
               self.d_embeddings.place_embeddings
    
    def forward(self,query_place_ids,
                     device='cpu'):
        batch_size=query_place_ids.size(0)
        # get embeddings of all places
        ## all_place_ids:(num_places,)
        all_place_ids=torch.arange(0,self.num_places,1,
                                   dtype=torch.long,
                                   device=device)

        ## outflow
        if self.optimize_out:
            out_query_places=self.o_embeddings(query_place_ids).\
                                    view(batch_size,
                                         1,
                                         self.place_embed_dim)
            # out_context_places:(1,num_places,place_embed_dim)->(batch_size,num_places,place_embed_dim)
            out_context_places=self.d_embeddings(all_place_ids).view(1,
                                                                     self.num_places,
                                                                     self.place_embed_dim)
            out_context_places=out_context_places.expand(batch_size,
                                                         -1,
                                                         -1)
            _,outflow_trans_weight=attn_places(out_query_places,
                                               out_context_places)
        else:
            outflow_trans_weight=None
        ## inflow
        if self.optimize_in:
            in_query_places=self.d_embeddings(query_place_ids).\
                                    view(batch_size,
                                         1,
                                         self.place_embed_dim)
            # in_context_places:(1,num_places,place_embed_dim)->(batch_size,num_places,place_embed_dim)
            in_context_places=self.o_embeddings(all_place_ids).view(1,
                                                                    self.num_places,
                                                                    self.place_embed_dim)
            in_context_places=in_context_places.expand(batch_size,
                                                       -1,
                                                       -1)
            _,inflow_trans_weight=attn_places(in_query_places,
                                              in_context_places)
        else:
            inflow_trans_weight=None
            
        return outflow_trans_weight,inflow_trans_weight
               
    def init_paras(self,):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)