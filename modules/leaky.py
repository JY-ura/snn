import torch
from torch import nn
from functional.surrogate import Sigmoid
import weakref


class LIF(nn.Module):
    
    lifs:weakref.WeakSet = weakref.WeakSet()
    
    def __init__(self, beta, u_th, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.u_th = u_th
        self.v = None
        self.lifs.add(self)
        
    def forward(self, input):
        if self.v is None:
            self.v = torch.zeros_like(input)
            
        self.v, output = _leaky_intergrate_and_fire(self.v, input, self.beta, self.u_th) 
        return output
    
    @classmethod
    def reset(cls):
        for lif in cls.lifs:
            lif.v = None
            

def _leaky_intergrate_and_fire(v, input, beta, u_th):
    """input current voltage, weighted spike input
    output updated voltage, spike output

    Args:
        v (_type_): _description_
        input (_type_): _description_
    """
    v = v*beta + input
    output = Sigmoid.apply(v - u_th)
    v -= output * u_th
    return v, output
    
