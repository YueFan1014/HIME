import torch


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1 + 1e-7)
        ctx.save_for_backward(x)
        z = x.double()
        #print(z)
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def arcosh(x):
    return Arcosh.apply(x)


def hyp_dist(x, y):
    temp = (x-y).norm(dim=-1, p=2, keepdim=True)
    temp_x = x.norm(dim=-1, p=2, keepdim=True)
    temp_y = y.norm(dim=-1, p=2, keepdim=True)
    return arcosh(1+2*temp*temp/((1-temp_x*temp_x)*(1-temp_y*temp_y)))
