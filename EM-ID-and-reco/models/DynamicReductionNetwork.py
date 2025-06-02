import torch
import torch.nn as nn
from .DynamicReductionNetworkJit import DynamicReductionNetworkJit
from .DynamicReductionNetworkOld import DynamicReductionNetworkOld
from .DynamicReductionNetworkTriton_v1 import DynamicReductionNetworkTriton_v1
from .DynamicReductionNetworkMustache import DynamicReductionNetworkMustache
from .DynamicReductionNetworkMustacheNoFlags import DynamicReductionNetworkMustacheNoFlags
from .DynamicReductionNetworkObject import DynamicReductionNetworkObject

class DynamicReductionNetwork(nn.Module):
    '''
    This model iteratively contracts nearest neighbour graphs 
    until there is one output node.
    The latent space trained to group useful features at each level
    of aggregration.
    This allows single quantities to be regressed from complex point counts
    in a location and orientation invariant way.
    One encoding layer is used to abstract away the input features.

    @param input_dim: dimension of input features
    @param hidden_dim: dimension of hidden layers
    @param output_dim: dimension of output
    
    @param k: size of k-nearest neighbor graphs
    @param aggr: message passing aggregation scheme. 
    @param norm: feature normaliztion. None is equivalent to all 1s (ie no scaling)
    @param loop: boolean for presence/absence of self loops in k-nearest neighbor graphs
    @param pool: type of pooling in aggregation layers. Choices are 'add', 'max', 'mean'
    
    @param agg_layers: number of aggregation layers. Must be >=0
    @param mp_layers: number of layers in message passing networks. Must be >=1
    @param in_layers: number of layers in inputnet. Must be >=1
    @param out_layers: number of layers in outputnet. Must be >=1
    '''
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, k=16, aggr='add', norm=None, 
                 loop=True, pool='max',
                 agg_layers=2, mp_layers=2, in_layers=1, out_layers=3,
                 graph_features = False,
                 latent_probe=None,
                 actually_jit=True,
                 kind = 'Object',
    ):
        super(DynamicReductionNetwork, self).__init__()
#        DRN = DynamicReductionNetworkJit
        if kind == 'Object':
            DRN = DynamicReductionNetworkObject
        elif kind == 'Mustache':
            DRN = DynamicReductionNetworkMustache
        elif kind == 'MustacheNoFlags':
            DRN = DynamicReductionNetworkMustacheNoFlags
        
        drn = DRN(
            input_dim_xECAL=input_dim,
            output_dim=output_dim,
        )

        if actually_jit:
            self.drn = torch.jit.script(drn)
        else:
            self.drn = drn

        self.kind = kind

    def forward(self, data):
        '''
        Push the batch 'data' through the network
        '''
        if isinstance(self.drn, DynamicReductionNetworkOld):
            return self.drn(data)
        if self.kind == 'Object':
            return self.drn(
                data.xECAL,
                data.fECAL,
                data.gainECAL,
                data.xES,
                data.fES,
                data.gx,
                data.xECAL_batch,
                data.xES_batch)
        elif self.kind == 'Mustache':
            return self.drn(
                data.xECAL,
                data.fECAL,
                data.gainECAL,
                data.gx,
                data.xECAL_batch)
        elif self.kind == 'MustacheNoFlags':
            return self.drn(
                    data.xECAL,
                    data.gx,
                    data.xECAL_batch)
