"""
Python module for holding our PyTorch models.
"""

from .EdgeNet import EdgeNet
from .EdgeNet2 import EdgeNet2
from .EdgeNetWithCategories import EdgeNetWithCategories
from .DynamicReductionNetwork import DynamicReductionNetwork
from .DynamicReductionNetworkTriton_v1 import DynamicReductionNetworkTriton_v1
from .gnn_geometric import GNNSegmentClassifier    
from .GravNet import GravNet, energy_fraction_loss, compressed_loss, abs_energy_fraction_loss
from training.classifier import classifier_loss

_models = {'EdgeNetWithCategories': EdgeNetWithCategories,
           'EdgeNet2': EdgeNet2,
           'EdgeNet': EdgeNet,
           'heptrkx_segment_classifier': GNNSegmentClassifier,
           'GravNet': GravNet,
           'DynamicReductionNetwork': DynamicReductionNetwork,
           'DynamicReductionNetworkTriton_v1': DynamicReductionNetworkTriton_v1,
}

_losses = {'energy_fraction_loss': energy_fraction_loss,
        'compressed_loss' : compressed_loss,
        'abs_energy_fraction_loss' : abs_energy_fraction_loss,
        "classifier_loss" : classifier_loss
}

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception('Model %s unknown' % name)

# attach custom losses to functional
from torch import nn
def get_losses():
    for loss,fn in _losses.items():
        setattr(nn.functional, loss, fn)
