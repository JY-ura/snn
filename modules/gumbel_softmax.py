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


argsoftmax = ArgSoftmax.apply


class GumbelSampler(nn.Module):
    def __init__(self, img, lamda, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter((1 - img)*0.9 + 0.05) 
        self.lamda = lamda
        
    def forward(self,):
        origianl_shape = self.alpha.shape
        
        alpha_flatten = self.alpha.flatten()
        alpha = torch.stack([alpha_flatten, 1-alpha_flatten], dim=-1)
        alpha_sample = (torch.log(alpha+1e-6) - torch.log(-torch.log(torch.rand_like(alpha)+1e-6))) / self.lamda
        
        img_sampled = argsoftmax(alpha_sample)
        img_sampled = img_sampled.reshape(origianl_shape)
        return img_sampled
    
    
if __name__ == '__main__':
    img = torch.tensor(
        [
            [0,1,0],
            [1,1,0],
            [0,0,1]
        ]
    )
    print(img)
    sampler = GumbelSampler(img, lamda=0.05)
    print(sampler())
        
        