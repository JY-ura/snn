from torch import autograd
from torch import nn
import torch

class ArgSoftmax(autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        """
        Args:
            ctx (_type_): _description_
            x (_type_): (**x_shape, 2)
            temperature (_type_): _description_

        Returns:
            _type_: _description_
        """
        prob = torch.softmax(x, dim=-1)
        ctx.save_for_backward(prob)
        return torch.argmax(x, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        """

        Args:
            ctx (_type_): _description_
            grad_output (_type_): (**x_shape,)

        Returns:
            _type_: **x_shape, 2
        """
        prob, = ctx.saved_tensors()
        prob_mask = (prob > 0.5).float()
        grad_output = torch.unsqueeze(grad_output, dim=-1).repeat_interleave(2, dim=-1) * prob_mask
        jac = prob @ prob.T
        jac = - jac + torch.diag(prob[0])
        return jac @ grad_output
    
    
class DiffArgmax(autograd.Function):
    """diffentialble argmax

    Args:
        autograd (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    @staticmethod
    def forward(ctx, x):
        assert len(x.shape) == 2, 'only accept tensor with two dim, with dim 2 indicating prob'
        y = torch.argmax(x, dim=-1).float()
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        grad_mask = torch.arange(0, 2) == y.unsqueeze(-1)
        grad = torch.zeros(y.shape + (2,))
        grad[grad_mask] = grad_output
        return grad


class GumbelSoftmax(nn.Module):
    
    def __init__(self, lamda: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lamda = lamda
        
        
    def forward(self, alpha):
        alpha_shape = alpha.shape
        # alpha = torch.flatten(alpha)
        u = torch.rand_like(alpha,)
        g = -torch.log(-torch.log(u + 1e-8))
        G = torch.softmax((torch.log(alpha) + g )/ self.lamda, dim=-1)
        output = DiffArgmax.apply(G)
        return output
    
    

if __name__ == '__main__':
    alpha = torch.tensor([
        [0.8,0.2],
        [0.3,0.7]
    ])
    alpha.requires_grad = True
    x = torch.tensor([0,1]).float()
    sampler = (0.1)
    optmizer = torch.optim.SGD((alpha,), 0.01)
    
    for _ in range(1000):
        y = sampler(alpha)
        loss = torch.sum((x-alpha)**2)
        loss.backward()
        optmizer.step()
        optmizer.zero_grad()
        print(loss)