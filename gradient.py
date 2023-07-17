from typing import Any
import torch
from torch.autograd import Function


class Fire(Function):
    def __init__(self):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def forward(ctx: Any, input,) -> Any:
        ctx.save_for_backward(input)
        return (input > 0).float()
    

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        input = ctx.saved_tensors[0]
        input_sigmoid = torch.sigmoid(input)
        input_gradient = input_sigmoid * (1-input_sigmoid)
        return grad_output * input_gradient
    
    
fire = Fire.apply

if __name__ == '__main__':
    x = torch.randn((10,10))
    x.requires_grad_()
    threshold = 0.1
    print('input of lif\n', x - threshold)
    output = fire(x-threshold)
    print('output of lif\n', output)
    
    torch.sum(output).backward()
    print('the gradient of x\n', x.grad)
    
    
    
    