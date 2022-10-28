import torch
import numpy as np
import torch.utils.data as data

class BOP(torch.optim.Optimizer):
    def __init__(self, params, gamma, tau):

        params_list = list(params)
        defaults = dict(lr=gamma, tau=tau)
        super(BOP, self).__init__(params_list, defaults)

        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(p.data.sign())
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.autograd.no_grad()
    def step(self):
        for group in self.param_groups:
            gamma = group["lr"]
            tau = group["tau"]
            for p in group['params']:
                if p.grad is None:
                    continue
                buff = self.state[p]['momentum_buffer']
                buff.mul_(1-gamma).add_(p.grad,alpha=gamma)
                swap = ((buff.abs() > tau).long()).mul((buff.sign() == p.sign()).long())
                #print(p)
                p.mul_(1 - 2 * swap)
                #print(p)
                #input('')






