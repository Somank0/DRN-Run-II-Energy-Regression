import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph, graclus_cluster
from torch_scatter import scatter
from torch_sparse.storage import SparseStorage

from torch import Tensor
from torch_geometric.typing import OptTensor, Optional, Tuple


from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import (max_pool, max_pool_x, global_max_pool,
                                avg_pool, avg_pool_x, global_mean_pool, 
                                global_add_pool)

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index[0], edge_index[1]
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

# jit compatible version of coalesce
def coalesce(index, value: OptTensor, m: int, n: int, op: str = "add"):
    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()

# jit compatible version of to_undirected
def to_undirected(edge_index, num_nodes: Optional[int] = None) -> Tensor:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    temp = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    row, col = temp[0], temp[1]
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

# jit compatible version of pool_edge, depends on coalesce
def pool_edge(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr

def _aggr_pool_x(cluster, x, aggr: str, size: Optional[int] = None):
    """Call into scatter with configurable reduction op"""
    return scatter(x, cluster, dim=0, dim_size=size, reduce=aggr)

def global_pool_aggr(x, batch: OptTensor, aggr: str, size: Optional[int] = None):
    """Global pool via passed aggregator: 'mean', 'add', 'max'"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    if batch is not None:
        size = int(batch.max().item() + 1)
    assert batch is not None
    return scatter(x, batch, dim=0, dim_size=size, reduce=aggr)

# this function is specialized compared to the more general non-jittable version
# in particular edge_attr can be removed since it is always None
def aggr_pool(cluster, x, batch: OptTensor, aggr: str) -> Tuple[Tensor, OptTensor]:
    """jit-friendly version of max/mean/add pool"""
    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)
    return x, batch

def aggr_pool_x(cluster, x, batch: OptTensor, aggr: str, size: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    """*_pool_x with configurable aggr method"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    if size is not None and batch is not None:
        batch_size = int(batch.max().item()) + 1
        return _aggr_pool_x(cluster, x, aggr, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)

    return x, batch
    
class DynamicReductionNetworkObject(nn.Module):
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
    @param output_dim: dimensio of output
    
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
    def __init__(self, input_dim_xECAL=5, input_dim_fECAL = 4,  input_dim_xES = 4, input_dim_fES = 1,
            hidden_dim=64, 
            output_dim=6, k=16, aggr='add', 
            loop=True, pool='max',
            agg_layers=6, mp_layers=3, in_layers=4, out_layers=2,
            graph_features = 2):
        super(DynamicReductionNetworkObject, self).__init__()

        self.graph_features = graph_features

        self.loop = loop

        self.k = k

        self.gain_embed = nn.Embedding(3,1, padding_idx=0)
        self.fECAL_embed = nn.Embedding(8, 3)

        self.fES_embed = nn.Embedding(2,1)

        #construct inputnet
        in_layers_ECAL_l = []
        in_layers_ECAL_l += [nn.Linear(input_dim_xECAL + input_dim_fECAL, hidden_dim),
                nn.ELU()]

        for i in range(in_layers-1):
            in_layers_ECAL_l += [nn.Linear(hidden_dim, hidden_dim), 
                    nn.ELU()]

        self.inputnetECAL = nn.Sequential(*in_layers_ECAL_l)

        #construct inputnet
        in_layers_ES_l = []
        in_layers_ES_l += [nn.Linear(input_dim_xES + input_dim_fES, hidden_dim),
                nn.ELU()]

        for i in range(in_layers-1):
            in_layers_ES_l += [nn.Linear(hidden_dim, hidden_dim), 
                    nn.ELU()]

        self.inputnetES = nn.Sequential(*in_layers_ES_l)

        #construct aggregation layers
        self.agg_layers = nn.ModuleList()
        for i in range(agg_layers):
            #construct message passing network
            mp_layers_l = []

            for j in range(mp_layers-1):
                mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                        nn.ELU()]

            mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ELU()]
           
            convnn = nn.Sequential(*mp_layers_l)
            
            self.agg_layers.append(EdgeConv(nn=convnn, aggr=aggr).jittable())

        #construct outputnet
        out_layers_l = []

        for i in range(out_layers-1):
            out_layers_l += [nn.Linear(hidden_dim+self.graph_features, hidden_dim+self.graph_features), 
                    nn.ELU()]

        out_layers_l += [nn.Linear(hidden_dim+self.graph_features, output_dim)]

        self.output = nn.Sequential(*out_layers_l)

        if pool not in {'max', 'mean', 'add'}:
            raise Exception("ERROR: INVALID POOLING")
        
        self.aggr_type = pool

    def forward(self, xECAL: Tensor, fECAL: Tensor, gain: Tensor, xES: Tensor, fES: Tensor, graph_x: Tensor, xECAL_batch: OptTensor, xES_batch: OptTensor) -> Tensor:
        '''
        Push the batch 'data' through the network
        '''
        #print('xECAL', xECAL.shape)
        #print('fECAL', fECAL.shape)
        fECAL = self.fECAL_embed(fECAL)
        gain = self.gain_embed(gain)
        ECAL = torch.cat( (xECAL, fECAL, gain), 1)
        #print('ECAL', ECAL.shape)
        #print('xES', xES.shape)
        #print('fES', fES.shape)
        fES = self.fES_embed(fES)
        ES = torch.cat( (xES, fES), 1)
        #print('ES', ES.shape)

        graph_x = graph_x.view((-1, self.graph_features))
        #print('GX', graph_x.shape)

        ECAL = self.inputnetECAL(ECAL)
        ES = self.inputnetES(ES)

        hits = torch.cat( (ECAL, ES), 0)
        #print('HITS', hits.shape)

        if xECAL_batch is not None and xES_batch is not None:
            batch = torch.cat( (xECAL_batch, xES_batch), 0)
        else:
            batch = None
        #print("BATCH", batch.shape)
        #print(batch)

        #batch tensor needs to be sorted, so we need to sort everything here
        #I hate this, but I guess it's just O(nlogn) so maybe that's okay
        #I wonder whether I can do faster sorting on my own
        #given that it's just the merge of two sorted tensors
        #which is O(n)
        if batch is not None:
            batch, sort_idxs = torch.sort(batch)
            hits = hits[sort_idxs]
        #print("SORTED")
        #print('HITS', hits.shape)
        #print("BATCH", batch.shape)
        #print(batch)

        # if there are no aggregation layers just leave x, batch alone
        nAgg = len(self.agg_layers)
        for i, edgeconv in enumerate(self.agg_layers):
            knn = knn_graph(hits, self.k, batch, loop=self.loop, flow=edgeconv.flow)
            edge_index = to_undirected(knn)
            hits = edgeconv(hits, edge_index)

            weight = normalized_cut_2d(edge_index, hits)
            cluster = graclus_cluster(edge_index[0], edge_index[1], weight, hits.size(0))

            if i == nAgg - 1:
                hits, batch = aggr_pool_x(cluster, hits, batch, self.aggr_type)
            else:
                hits, batch = aggr_pool(cluster, hits, batch, self.aggr_type)
            
            #print("AGGR",i,hits.shape, batch.shape)
            #print(batch)
        
        # this xforms to batch-per-row so no need to return batch
        hits = global_pool_aggr(hits, batch, self.aggr_type)
        #print("END AGGR", hits.shape)

        if graph_x is not None:
            hits = torch.cat((hits, graph_x), 1)

        hits = self.output(hits).squeeze(-1)

        return hits
