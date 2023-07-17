from typing import Any
import torch 
from torch import autograd


class Sigmoid(autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def forward(ctx: Any, input) -> Any:
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    
    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        input = ctx.saved_tensors[0]
        input_sigmoid = torch.sigmoid(input)
        input_grad = input_sigmoid * (1-input_sigmoid)
        return grad_outputs * input_grad