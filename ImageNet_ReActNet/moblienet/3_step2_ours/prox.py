import torch
import copy
import math

def binarise_sym(real_weights):
    scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
    return scaling_factor * torch.sign(real_weights)

class prox_opt(object):
    def __init__(self, model, reg, epochs, freeze, summary_writer):
        print('setting up prox')

        self.inital_reg = float(reg)
        self.max_epochs = epochs
        self.summary_writer = summary_writer
        self.freeze = freeze
        self.frozen = False
        self.epoch = 120

        self.select_binary_params(model)
        self.adjust_reg()
        self.calc_mean_dist(quant=False)

    def adjust_reg(self, max_reg=1.0):
        new_reg = float(min(self.inital_reg * (self.epoch+1) / self.max_epochs, max_reg))
        self.reg = copy.copy(new_reg)
        print('updating reg, new value:', new_reg)

    def post_update(self):
        with torch.no_grad():
            if self.reg >= 0:
                for name, p in self.binary_params.items():
                    p.data.copy_((p.data + 2 * p.data * self.reg).clamp(min=-1, max=1))

    def _epoch(self, epoch):
        self.epoch = epoch
        self.adjust_reg()
        if self.epoch > self.freeze:
            self.binary_freeze()
        self.print_mean_dist()

    def binary_freeze(self):
        print('freezing binary params')
        self.frozen = True
        for name, p in self.binary_params.items():
            with torch.no_grad():
                p.data.copy_(binarise_sym(p.data))
                p.requires_grad = False

    def select_binary_params(self, model): 
        self.binary_params = {}
        for n, param in model.named_parameters():
            if 'binary' in n:
                self.binary_params[n] = param
        print('binary params: ')
        for name in self.binary_params:
            p = self.binary_params[name]
            print(name, p.size())

    def calc_mean_dist(self, quant):
        demon = 0
        num = 0
        min_w =  1e6
        max_w = -1e6
        postive_weights = 0
        negative_weights = 0
        postive_mean = 0 
        negative_mean = 0
        postive_squared_mean = 0
        negative_squared_mean = 0
        zeros = 0
        for name, p in self.binary_params.items():
            with torch.no_grad():
                num += (p.data - p.data.sign()).abs().sum()
                demon += p.data.nelement()
                max_w = p.data.max() if p.data.max() > max_w else max_w
                min_w = p.data.min() if p.data.min() < min_w else min_w
                pos_inds = (p>0)
                neg_inds = (p<0)
                zero = (p==0)
                postive_weights += pos_inds.long().sum()
                negative_weights += neg_inds.long().sum()
                zeros += zero.long().sum()
                pos = p[pos_inds]
                neg = p[neg_inds]
                postive_mean += pos.sum()
                negative_mean += neg.sum()
                postive_squared_mean += pos.pow(2).sum()
                negative_squared_mean += neg.pow(2).sum()
        if demon == 0:
            raise RuntimeError('no binary weights found')
        mean_dist = num/demon
        self.mean_dist=float(mean_dist.item())
        self.max_w=float(max_w.item())
        self.min_w=float(min_w.item())
        self.num_pos = int(postive_weights.item())  
        self.num_neg = int(negative_weights.item())  
        self.zeros = int(zeros.item())  
        self.pos_mean = float(postive_mean/postive_weights)
        self.neg_mean = float(negative_mean/negative_weights)
        self.pos_std = float((postive_squared_mean/postive_weights-self.pos_mean**2).sqrt())
        self.neg_std = float((negative_squared_mean/negative_weights-self.pos_mean**2).sqrt())
        self.w_norm = float((postive_squared_mean+negative_squared_mean).sqrt())

    def print_mean_dist(self, quantised=False):
        self.calc_mean_dist(quant=quantised)
        print('dist: {dist_:.6f} || max w: {max_:.4f} || '
              'min w: {min__:.4f} || reg: {reg_:.6f} || '
              'neg mean {neg_mean__:.4f} || pos mean: {pos_mean__:.4f} || '
              'neg std {neg_std__:.4f} || pos std: {pos_std__:.4f} || zeros: {zero__:.4f}'.format(
              dist_=self.mean_dist, 
              max_=self.max_w, 
              min__=self.min_w, 
              reg_=self.reg,
              pos_mean__=self.pos_mean,
              neg_mean__=self.neg_mean,
              pos_std__=self.pos_std,
              neg_std__=self.neg_std,
              zero__=self.zeros,
              ))
        print('TOTAL',self.num_pos+self.num_neg)
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('float/mean_dist', float(self.mean_dist), self.epoch)
            self.summary_writer.add_scalar('float/max_w', float(self.max_w), self.epoch)
            self.summary_writer.add_scalar('float/min_w', float(self.min_w), self.epoch)
            self.summary_writer.add_scalar('float/w_norm', float(self.w_norm), self.epoch)
            self.summary_writer.add_scalar('float/pos_mean', float(self.pos_mean), self.epoch)
            self.summary_writer.add_scalar('float/pos_std', float(self.pos_std), self.epoch)
            self.summary_writer.add_scalar('float/neg_mean', float(self.neg_mean), self.epoch)
            self.summary_writer.add_scalar('float/neg_std', float(self.neg_std), self.epoch)
