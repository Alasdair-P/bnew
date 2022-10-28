"""Popular Learning Rate Schedulers"""
from __future__ import division
import math


class LRScheduler(object):
    r"""Learning Rate Scheduler

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """

    def __init__(self, mode, base_lr=0.01, target_lr=0, niters=0, nepochs=0, iters_per_epoch=0,
                 offset=0, power=2, step_iter=None, step_epoch=None, step_factor=0.1, min_lr=1e-6):
        super(LRScheduler, self).__init__()
        assert (mode in ['constant', 'step', 'linear', 'inverse_linear', 'poly', 'inverse_poly', 'cosine'])

        self.mode = mode
        if mode == 'step':
            assert (step_iter is not None or step_epoch is not None)
        self.base_lr = base_lr
        self.target_lr = target_lr
        if self.mode == 'constant':
            self.target_lr = self.base_lr

        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s * iters_per_epoch for s in step_epoch]

        self.offset = offset
        self.power = power
        self.step_factor = step_factor
        self.min_lr=min_lr
        self.decay_factor = 1.0

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'inverse_linear':
            factor = T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'inverse_poly':
            factor = pow(T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + math.cos(math.pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor
        self.learning_rate = self.learning_rate * self.decay_factor + (1-self.decay_factor) * self.min_lr

    def update_lr(self, optimizer, cur_iter):
        self.update(cur_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def decay_lr(self, optimizer, cur_iter):
        for param_group in optimizer.param_groups:
            self.decay_factor *= 0.1
            print('decaying learing rate by factor of 10')


if __name__ == '__main__':
    lr_scheduler = LRScheduler(mode='poly', base_lr=0.00045, nepochs=100,
                               iters_per_epoch=371, power=0.9)
    for i in range(60 * 176):
        lr = lr_scheduler(i)
        print(lr)
    input('')
    lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)
