import torch
import copy
import math
import matplotlib.pyplot as plt

with torch.no_grad():
    def ternarize(A, delta_method='mean'):
        '''Ternarize a Float tensor'''
        A_quant = A.clone()
        if delta_method == 'max':
            delta = A.abs().max() * 0.05
        elif delta_method == 'mean':
            delta = A.abs().mean() * 0.7
            #delta = A.abs().mean() * 0.4
        else:
            print(not_valid_delta_method)
        A_quant.masked_fill_(A.abs() < delta, 0)
        inds_p, inds_n = (A >= delta), (A <= -delta)
        A_quant.masked_fill_(inds_p, A[inds_p].mean())
        A_quant.masked_fill_(inds_n, A[inds_n].mean())
        return A_quant

with torch.no_grad():
    def ternarize_symetric(A):
        A_quant = A.clone()
        delta = A.abs().mean() * 0.7
        A_quant.masked_fill_(A.abs() < delta, 0)
        inds_p, inds_n = (A >= delta), (A <= -delta)
        num = A[inds_p].abs().sum() + A[inds_n].abs().sum()
        denom = A[inds_p].nelement() + A[inds_n].nelement()
        value = num/denom
        A_quant.masked_fill_(inds_p,  value)
        A_quant.masked_fill_(inds_n, -value)
        return A_quant

def binarise_sym(real_weights):
    scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
    return scaling_factor * torch.sign(real_weights)

def w_reg(p, reg):
    with torch.no_grad():
        p.data.copy_((p.data + p.data.sign() * reg).clamp(min=-1, max=1))

def w_reg_with_scalar(p, reg):
    with torch.no_grad():
        a = p.data - binarise_sym(p.data)
        p.data.copy_((p.data - (a.abs().clamp(max=reg) * a.sign())).clamp(min=-1,max=1))

def cosine_reg(p, reg):
    with torch.no_grad():
        w = p.data
        q = binarise_sym(w.data)
        w_ = (w**2).sum(dim=(1,2,3)).sqrt()
        q_ = (q**2).sum(dim=(1,2,3)).sqrt()
        grad = q/(w_*q_) - w*(w*q).sum(dim=(1,2,3))/(q_*w_**3)
        p.data.copy_(p.data - reg * grad)

def _u_reg(p, reg):
    with torch.no_grad():
        p.data.copy_((p.data + 4 * p.data**3 * reg).clamp(min=-1, max=1))

def u_reg(p, reg):
    with torch.no_grad():
        p.data.copy_((p.data + 2 * p.data * reg).clamp(min=-1, max=1))

def U_reg(p, reg):
    with torch.no_grad():
        p.data.copy_(((p.data + p.data.sign() * reg)/(1 + reg)).clamp(min=-1, max=1))

def ternary_reg(p, reg):
    with torch.no_grad():
        p.data.copy_((p.data + ternarize(p.data) * reg)/(1 + reg))

def convex_hull(p, reg):
    with torch.no_grad():
        p.data.copy_(p.clamp(min=-1, max=1))

def ternary_reg_l05(p, reg):
    with torch.no_grad():
        w_wq = p.data - ternarize(p.data)
        p.data.copy_(p.data - ((reg*(w_wq.abs() ** -0.5)).clamp(max=w_wq.abs())).mul(w_wq.sign()))

def ternary_reg_l1(p, reg):
    with torch.no_grad():
        w_wq = p.data - ternarize(p.data)
        p.data.copy_(p.data - (reg*w_wq.sign()))

def tanh_proj(p, reg, levels, const):
    with torch.no_grad():
        if levels == 2:  # binary
            p.data.copy_(torch.tanh(p.data * reg))
        elif levels == 3:    # ternary, shifted tanh
            p.data.copy_(0.5 * (torch.tanh(reg*(p.data*const + 0.5)) + torch.tanh(reg*(p.data*const - 0.5))))
        else:
            assert(0)

def hard_proj(p, reg_type):
    with torch.no_grad():
        if reg_type == 'bnew_binary_w_s':
            p_out = binarise_sym(p.data)
        elif 'ste_binary_w_s' in reg_type:
            p_out = binarise_sym(p.data)
        elif 'binary' in reg_type:
            p_out = p.data.sign()
        elif 'bnew' in reg_type:
            p_out = p.data.sign()
        elif reg_type == 'mirror_ternary':
            p_out = p.clone() 
            p_out.data[p.data.le(-0.5)] = -1
            p_out.data[p.data.ge(0.5)] = 1
            p_out.data[p.data.lt(0.5)*p.data.gt(-0.5)] = 0
        elif 'ternary' in reg_type:
            p_out = ternarize(p.data)
        elif reg_type == 'none':
            p_out = p.data.sign()
            pass
        elif reg_type == 'convex_hull':
            p_out = p.data.sign()
        else:
            raise RuntimeError('reg type not found')
    return p_out

class prox_opt(object):
    def __init__(self, model, args, summary_writer):
        self.binary_params = None
        self.args = args
        self.summary_writer = summary_writer
        self.frozen = False
        print('setting up prox')
        self.reg_type = args.reg_type
        self.w_dict = {'floating_point_params' : ['conv1.weight', 'fc.weight']}  
        self.select_binary_params(model)
        self.inital_reg = float(args.reg)
        self.max_epochs = args.epochs
        self.epoch = 0.0
        self.adjust_reg()
        if 'mirror' in self.reg_type or 'ste' in self.reg_type or self.args.quant_val or args.eval:
            print('setting up aux weights')
            self.set_up_saved_params()
        if 'mirror' in self.reg_type:
            self.mirror_const = 1/(0.5*self.reg*(2-math.tanh(self.reg*0.5)**2-math.tanh(-self.reg*0.5)**2)) 
        self.calc_mean_dist(quant=False)

    def adjust_reg(self, max_reg=1.0):
        if self.binary_params:
            if not 'mirror' in self.args.reg_type:
                new_reg = float(min(self.inital_reg * (self.epoch+1) / self.args.epochs, max_reg))
                self.reg = copy.copy(new_reg)
            else:
                new_reg = float(min((self.inital_reg**self.epoch), 10**35))
                self.reg = copy.copy(new_reg)
            print('updating reg, new value:', new_reg)

    def print_first_binary_layer_weights(self):
        with torch.no_grad():
            for name, p in self.binary_params.items():
                print('p',p.reshape(-1)[:20])
                print('p (saved)',self.saved_params[name].reshape(-1)[:20])
                if not p.grad == None:
                    print('grad',p.grad.reshape(-1)[:20])
                print(self.reg)
                input('')
                return

    def soft_project_params(self):
        with torch.no_grad():
            if self.binary_params and self.reg >= 0:
                reg_type = self.reg_type
                reg = self.reg
                if reg_type == 'l1_binary':
                    for name, p in self.binary_params.items():
                        w_reg(p, reg)
                elif reg_type == 'l2_binary':
                    for name, p in self.binary_params.items():
                        U_reg(p, reg)
                elif reg_type == 'bnew_binary':
                    for name, p in self.binary_params.items():
                        u_reg(p, reg)
                elif reg_type == 'cosine_reg':
                    for name, p in self.binary_params.items():
                        cosine_reg(p, reg)
                elif reg_type == 'bnew_binary_w_s':
                    for name, p in self.binary_params.items():
                        w_reg_with_scalar(p, reg)
                elif reg_type == 'ternary':
                    for name, p in self.binary_params.items():
                        ternary_reg(p, reg)
                elif reg_type == 'convex_hull':
                    for name, p in self.binary_params.items():
                        convex_hull(p, reg)
                elif reg_type == 'none':
                    pass
                elif reg_type == 'bnew':
                    pass
                elif reg_type == 'bop_binary':
                    pass
                elif reg_type == 'bnew_ternary':
                    for name, p in self.binary_params.items():
                        self.ternary_reg(p, name, reg)
                elif reg_type == 'l05_ternary':
                    for name, p in self.binary_params.items():
                        ternary_reg_l05(p, reg)
                elif reg_type == 'l1_ternary':
                    for name, p in self.binary_params.items():
                        ternary_reg_l1(p, reg)
                elif reg_type == 'mirror_ternary':
                    for name, p in self.binary_params.items():
                        self.saved_params[name].copy_(p.data)
                        tanh_proj(p, self.reg, 3, self.mirror_const)
                elif reg_type == 'mirror_binary':
                    for name, p in self.binary_params.items():
                        self.saved_params[name].copy_(p.data)
                        tanh_proj(p, self.reg, 2, self.mirror_const)
                elif 'ste' in reg_type:
                    pass
                else:
                    raise RuntimeError('reg type not found')

    def hard_project_params(self):
        if self.binary_params:
            print('projecting to quantised set')
            for name in self.binary_params:
                p = self.binary_params[name]
                with torch.no_grad():
                    p.data.copy_(hard_proj(p, self.reg_type))

    def pre_update(self):
        with torch.no_grad():
            if self.binary_params:
                if 'ste' in self.reg_type:
                    for name, p  in self.binary_params.items():
                        self.saved_params[name].copy_(p.data.clone().clamp(min=-1,max=1))
                        p.data.copy_(hard_proj(p, self.reg_type))
                elif self.reg_type == 'mirror_ternary':
                    for name, p  in self.binary_params.items():
                        self.saved_params[name].copy_(p.data.clone())
                        p.data.copy_(0.5 * (torch.tanh(self.reg*(p.data*self.mirror_const + 0.5)) + torch.tanh(self.reg*(p.data*self.mirror_const - 0.5))))
                elif self.reg_type == 'mirror_binary':
                    for name, p  in self.binary_params.items():
                        self.saved_params[name].copy_(p.data.clone())
                        p.data.copy_(torch.tanh(p.data * self.reg))
                elif self.reg_type == 'mirror_binary_clamp':
                    for name, p  in self.binary_params.items():
                        self.saved_params[name].copy_(p.data.clone().clamp(min=-1,max=1))
                        p.data.copy_(torch.tanh(p.data * self.reg))
                else:
                    pass


    def post_update(self):
        with torch.no_grad():
            if self.binary_params:
                if 'ste' in self.reg_type or 'mirror' in self.reg_type:
                     with torch.no_grad():
                        for name, p  in self.binary_params.items():
                            p.data.copy_(self.saved_params[name].clone())
                else:
                    self.soft_project_params()

    def ste_gradient(self):
        if self.binary_params:
            for name, p in self.binary_params.items():
                with torch.no_grad():
                    p.grad[p.data.abs().ge(1)] = 0 

    def _epoch(self, epoch):
        self.epoch = epoch
        self.adjust_reg()
        if self.epoch in self.args.freeze_epoch and self.args.freeze_epoch:
            self.binary_freeze()

    def binary_freeze(self):
        print('freezing binary params')
        if self.binary_params:
            self.frozen = True
            for name in self.binary_params:
                p = self.binary_params[name]
                with torch.no_grad():
                    p.data.copy_(hard_proj(p.data,self.reg_type))
                    p.requires_grad = False

    def ternary_reg(self, p, name, reg):
        with torch.no_grad():
            mask = self.masks[name]
            reg_mat = ((torch.ones_like(p)) * reg + mask).clamp(min=0,max=1)
            p.data.copy_((p.data * (1-reg_mat) + ternarize(p.data) * reg))

    def clamp_to_convex_hull(self):
        if self.binary_params:
            for name in self.binary_params:
                p = self.binary_params[name]
                with torch.no_grad():
                    p.data.clamp_(min=-1, max=1)

    def print_binary_params(self, model):
        if self.binary_params:
            for name, param in model.named_parameters():
                if (len(param.size()) == 1) or name in self.floating_point_params:
                    print('float', name, param.size())
                else:
                    print('binary', name, param.size())

    def select_binary_params(self, model): 
        self.binary_params = {}
        if not self.args.model == 'binary_fast_scnn':
            for n, param in model.named_parameters():
                if (len(param.size()) == 1) or n in self.w_dict['floating_point_params'] or 'bias' in n or 'downsample' in n:
                    pass
                else:
                    self.binary_params[n] = param
        else:
            for n, param in model.named_parameters():
                if 'binary' in n:
                    self.binary_params[n] = param

        print('binary params: ')
        for name in self.binary_params:
            p = self.binary_params[name]
            print(name, p.size())


    def set_up_saved_params(self):
        self.saved_params = {}
        if self.binary_params:
            for name in self.binary_params:
                p = self.binary_params[name]
                self.saved_params[name] = torch.zeros_like(p)

    def save_params(self):
        if self.binary_params:
            with torch.no_grad():
                for name in self.binary_params:
                    p = self.binary_params[name]
                    self.saved_params[name].copy_(p.data)

    def restore(self):
        if self.binary_params:
            with torch.no_grad():
                for name in self.binary_params:
                    p = self.binary_params[name]
                    p.data.copy_(self.saved_params[name])
 
    def calc_mean_dist(self, quant):
        if self.binary_params:
            demon = 0
            num = 0
            min_w =  1e6
            max_w = -1e6
            zeros = 0
            pos_ones = 0
            neg_ones = 0
            postive_weights = 0
            negative_weights = 0
            postive_mean = 0 
            negative_mean = 0
            postive_squared_mean = 0
            negative_squared_mean = 0
            for name, p in self.binary_params.items():
                with torch.no_grad():
                    num += (p.data - hard_proj(p.data, self.reg_type)).abs().sum()
                    demon += p.data.nelement()
                    max_w = p.data.max() if p.data.max() > max_w else max_w
                    min_w = p.data.min() if p.data.min() < min_w else min_w
                    pos_inds = (p>0)
                    neg_inds = (p<0)
                    zeros += (p==0).long().sum()
                    pos_ones += (p==1).long().sum()
                    neg_ones += (p==-1).long().sum()
                    postive_weights += pos_inds.long().sum()
                    negative_weights += neg_inds.long().sum()
                    pos = p[pos_inds]
                    neg = p[neg_inds]
                    postive_mean += pos.sum()
                    negative_mean += neg.sum()
                    postive_squared_mean += pos.pow(2).sum()
                    negative_squared_mean += neg.pow(2).sum()
                    if self.summary_writer:
                        if quant:
                            summary_tag = 'binary'
                        else:
                            summary_tag = 'float'
                        #self.summary_writer.add_histogram(summary_tag+'/'+name, p.clone().detach().cpu().numpy(), self.epoch)
            if demon == 0:
                raise RuntimeError('no binary weights found')
            mean_dist = num/demon
            self.mean_dist=float(mean_dist.item())
            self.max_w=float(max_w.item())
            self.min_w=float(min_w.item())
            self.num_pos = int(postive_weights.item())  
            self.num_neg = int(negative_weights.item())  
            self.zeros = int(zeros.item())  
            self.pos_ones = int(pos_ones.item())  
            self.neg_ones = int(neg_ones.item())  
            self.pos_mean = float(postive_mean/postive_weights)
            self.neg_mean = float(negative_mean/negative_weights)
            self.pos_std = float((postive_squared_mean/postive_weights-self.pos_mean**2).sqrt())
            self.neg_std = float((negative_squared_mean/negative_weights-self.pos_mean**2).sqrt())
            self.w_norm = float((postive_squared_mean+negative_squared_mean).sqrt())


    def print_mean_dist(self, quantised=False):
        if self.binary_params:
            self.calc_mean_dist(quant=quantised)
            print('-'*30)
            print('dist: {dist_:.6f} || max w: {max_:.4f} || '
                  'min w: {min__:.4f} || reg: {reg_:.6f} || Total: {total__} ||'
                  'zeros: {zeros__} || num neg w {num_neg__} || num pos w: {num_pos__} || '
                  'neg mean {neg_mean__:.4f} || pos mean: {pos_mean__:.4f} || '
                  'neg std {neg_std__:.4f} || pos std: {pos_std__:.4f} || '.format(
                  dist_=self.mean_dist, 
                  max_=self.max_w, 
                  min__=self.min_w, 
                  reg_=self.reg,
                  zeros__=self.zeros,
                  num_neg__=self.num_neg,
                  num_pos__=self.num_pos,
                  pos_mean__=self.pos_mean,
                  neg_mean__=self.neg_mean,
                  pos_std__=self.pos_std,
                  neg_std__=self.neg_std,
                  total__=self.num_pos+self.num_neg,
                  ))
            print('-'*30)
            if self.summary_writer is not None:
                if quantised:
                    self.summary_writer.add_scalar('binary/mean_dist', float(self.mean_dist), self.epoch)
                    self.summary_writer.add_scalar('binary/max_w', float(self.max_w), self.epoch)
                    self.summary_writer.add_scalar('binary/_min_w', float(self.min_w), self.epoch)
                    self.summary_writer.add_scalar('binary/zeros', float(self.zeros), self.epoch)
                    self.summary_writer.add_scalar('binary/pos_ones', float(self.pos_ones), self.epoch)
                    self.summary_writer.add_scalar('binary/neg_ones', float(self.neg_ones), self.epoch)
                else:
                    self.summary_writer.add_scalar('float/mean_dist', float(self.mean_dist), self.epoch)
                    self.summary_writer.add_scalar('float/max_w', float(self.max_w), self.epoch)
                    self.summary_writer.add_scalar('float/min_w', float(self.min_w), self.epoch)
                    self.summary_writer.add_scalar('float/num_pos', float(self.num_pos), self.epoch)
                    self.summary_writer.add_scalar('float/num_neg', float(self.num_neg), self.epoch)
                    self.summary_writer.add_scalar('float/w_norm', float(self.w_norm), self.epoch)

                    self.summary_writer.add_scalar('float/pos_mean', float(self.pos_mean), self.epoch)
                    self.summary_writer.add_scalar('float/pos_std', float(self.pos_std), self.epoch)
                    self.summary_writer.add_scalar('float/neg_mean', float(self.neg_mean), self.epoch)
                    self.summary_writer.add_scalar('float/neg_std', float(self.neg_std), self.epoch)

