import os
import argparse
import time
import shutil
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import mlogger
import pickle
from torchvision import transforms
from models import get_model
from utils.lr_scheduler import LRScheduler
from utils.logger import setup_xp
from utils.save import save_checkpoint
from optim import get_optimizer
from cli import parse_args
from loaders import get_data_loaders
from prox import prox_opt
from datetime import datetime
from tensorboardX import SummaryWriter
from test import get_test_func, train_acc
from KL_loss import DistributionLoss
from bop import BOP
import torchvision.models as models
import copy


class Trainer(object):
    def __init__(self, args):
        self.args = args
        assert 'bop' in args.reg_type

        if args.use_tensorboard:
            tb_log_dir = os.path.join(args.tb_logs, args.xp_name_no_dir) 
            print('tb_log', tb_log_dir)
            if not os.path.exists(tb_log_dir):
                os.makedirs(tb_log_dir)
            self.summary_writer = SummaryWriter(tb_log_dir)
        else:
            self.summary_writer = None

        # create data loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args)

        # create network
        self.model = get_model(args)

        self.stats = {}
        self.metrics = ['losses', 'train_acc', 'val_acc', 'dist_to_binary']
        for metric in self.metrics:
            self.stats[metric] = []

        # create evalution function
        self.test_func = get_test_func(args)
        self.train_acc = train_acc(args)

        # create teacher
        self.loss_func = torch.nn.CrossEntropyLoss().to(args.device)

        student_model = copy.copy(args.model)
        student_model_path = copy.copy(args.load_model)

        args.load_model = args.teacher
        args.model = 'rn'

        self.model_teacher = None

        args.model = student_model
        args.load_model = student_model_path

        # create qunatiser
        self.reg = prox_opt(self.model, self.args, self.summary_writer)
        self.best_pred = 0.0
        self.best_quant = 0.0

        # optimizer
        all_parameters = self.model.parameters()

        float_parameters = []
        for n, p in self.model.named_parameters():
            if (len(p.size()) == 1) or n in ['conv1.weight', 'fc.weight'] or 'bias' in n or 'downsample' in n:
                float_parameters.append(p)
                #print(n)
        float_parameters_id = list(map(id, float_parameters))
        binary_parameters = list(filter(lambda p: id(p) not in float_parameters_id, all_parameters))
        #input('')


        self.optimizer_b = BOP(binary_parameters, gamma=1e-4,tau=1e-8)
        self.optimizer_f = torch.optim.Adam(float_parameters, lr=args.lr, weight_decay=args.weight_decay)

        self.lr_scheduler_b = LRScheduler(mode='linear', base_lr=1e-4, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9, step_epoch = args.T)

        self.lr_scheduler_f = LRScheduler(mode='linear', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9, step_epoch = args.T)

    def train(self):
        cur_iter = 0
        start_time = time.time()

        epoch = 0

        #self.validation(epoch)
        #self.quant_validation(epoch, self.reg.soft_project_params)
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            self.loss_avg=0

            try: 
                if epoch in args.D:
                    self.lr_scheduler.decay_lr(self.optimizer, cur_iter)

                for i, (images, targets) in enumerate(self.train_loader):
                    images = images.to(self.args.device)
                    targets = targets.to(self.args.device)

                    self.lr_scheduler_b.update_lr(self.optimizer_b, cur_iter)
                    self.lr_scheduler_f.update_lr(self.optimizer_f, cur_iter)

                    self.reg.pre_update()

                    logits_student = self.model(images)
                    outputs = logits_student
                    loss = self.loss_func(logits_student, targets)

                    self.train_acc.update(i, outputs, targets)

                    self.optimizer_f.zero_grad()
                    self.optimizer_b.zero_grad()

                    loss.backward()
                    self.loss_avg += float(loss*len(targets))

                    self.reg.post_update()

                    self.optimizer_f.step()
                    self.optimizer_b.step()
                    
                    cur_iter += 1
                    if cur_iter % 20 == 0:
                        if not self.reg.binary_params:
                            print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                            epoch, args.epochs, i + 1, len(self.train_loader),
                            time.time() - start_time, self.lr_scheduler.learning_rate, loss.item()))
                        else:
                            print('Epoch: [{e_}/{E_}] Iter [{iter_}/{I_}] || '
                                  'Time: {time_:.4f} sec || lr: {lr_:.8f} || '
                                  'Loss: {loss_:.4f} || dist: {dist_:.6f} || '
                                  'reg: {reg_:.6f}'.format(e_=epoch, 
                              E_=args.epochs,
                              iter_=i + 1,
                              I_=len(self.train_loader),
                              time_=time.time() - start_time,
                              lr_=self.lr_scheduler_b.learning_rate,
                              loss_=loss.item(), 
                              dist_=self.reg.mean_dist, 
                              reg_=self.reg.reg))

            except KeyboardInterrupt:
                print('_' * 80)
                print('Exit form training early')
                save_checkpoint(self.model, self.optimizer, self.args, is_best=False)
                return


            tacc_1, tacc_2 = self.train_acc.return_acc()
            # every epoch
            self.reg._epoch(epoch)
            if self.args.no_val: # save every epoch
                save_checkpoint(self.model, self.optimizer, self.args, is_best=False)
            else:
                self.validation(epoch, None, self.val_loader)
                if self.args.quant_val:
                    self.validation(epoch, self.reg.hard_project_params, self.val_loader)

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('opt/train_acc', float(tacc_1), epoch)
                self.summary_writer.add_scalar('opt/loss', float(loss), epoch)
                self.summary_writer.add_scalar('opt/reg', float(self.reg.reg), epoch)
                self.summary_writer.add_scalar('opt/lr', float(self.lr_scheduler_b.learning_rate), epoch)

            self.stats['train_acc'].append(round(float(tacc_1),2))
            self.stats['val_acc'].append(round(float(self.metric_1),2))
            self.stats['dist_to_binary'].append(round(float(self.reg.mean_dist),4))
            self.stats['losses'].append(round(float(self.loss_avg/self.args.train_size),4))

        save_checkpoint(self.model, self.optimizer_b, self.args, is_best=False)

    @torch.autograd.no_grad()
    def validation(self, epoch, quant_function, loader):

        if quant_function is None:
            quant = 'float'
            is_quant = False
        else:
            quant = 'binary'
            self.reg.save_params()
            quant_function()
            is_quant = True

        self.reg.print_mean_dist(quantised=is_quant)
        print('evalulating model, quantise :', is_quant)
        self.metric_1, self.metric_2 = self.test_func(self.model, loader, epoch, self.args)

        if is_quant:
            self.reg.restore()

        if self.summary_writer is not None:
            if self.args == 'citys':
                self.summary_writer.add_scalar(quant + '/pixAcc',   float(self.metric_1), epoch)
                self.summary_writer.add_scalar(quant + '/mIoU',     float(self.metric_2), epoch)
            else:
                self.summary_writer.add_scalar(quant + '/Acc',      float(self.metric_1), epoch)
                self.summary_writer.add_scalar(quant + '/Acc_top5', float(self.metric_2), epoch)

        if is_quant:
            if self.metric_1 > self.best_quant:
                self.best_quant = self.metric_1
                save_checkpoint(self.model, self.optimizer_b, self.args, is_best=True, is_quantised=is_quant)
                print('saving new best qunatised model')
        else:
            if self.metric_1 > self.best_pred:
                self.best_pred = self.metric_1
                save_checkpoint(self.model, self.optimizer_b, self.args, is_best=True, is_quantised=is_quant)
                print('saving new best model')

    @torch.autograd.no_grad()
    def create_summary(self, this_name, tag, is_quant):
        if is_quant:
            tag = tag + '_binary'
            best = self.best_quant
        else:
            best = self.best_pred
        save_path = '/home/aparen/quantized-scnn/projects/pytorch_fscnn/results/'
        save_file = self.args.dataset + self.args.model + '_summary.txt'
        save_path = os.path.join(save_path, save_file)
        with open(save_path, 'a') as results:
            results.write('{name},{dataset},best,{miou:.4f}\n'
                    .format(name=this_name,
                            dataset=tag,
                            miou=best))
                    
    @torch.autograd.no_grad()
    def load_best_model(self,is_quant):
        self.create_summary(self.args.xp_name, 'val', is_quant)
        if is_quant:
            args.load_model =  '{}/best_quant_model.pkl'.format(args.xp_name)
            quant_func = self.reg.hard_project_params
        else:
            args.load_model =  '{}/best_model.pkl'.format(args.xp_name)
            quant_func = None
        self.model = get_model(args)
        self.reg.select_binary_params(self.model)
        self.validation(self.args.epochs, quant_func, self.val_loader)
        if is_quant:
            self.best_quant = 0.0
        else:
            self.best_pred = 0.0
        self.validation(self.args.epochs, quant_func, self.test_loader)
        self.create_summary(args.xp_name, 'test', is_quant)


if __name__ == '__main__':
    args = parse_args()
    cudnn.benchmark = True
    trainer = Trainer(args)
    with mlogger.stdout_to("{}/log.txt".format(args.xp_name), enabled=not(args.debug)):
        print(args)
        if args.eval:
            print('Evaluation model: ', args.resume)
            
            if False:
                print('quantised model')
                quant_func = trainer.reg.hard_project_params
                is_quant = True
            else:
                print('floating model')
                quant_func = None
                is_quant = False

            trainer.validation(args.start_epoch, quant_func, trainer.test_loader)
            trainer.create_summary(args.resume, 'test', is_quant)

        else:
            print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
            trainer.train()
            trainer.load_best_model(False)
            print(trainer.stats)
            a_file = open(args.xp_name+"/stats_for_plots.pkl", "wb")
            pickle.dump(trainer.stats, a_file)
            a_file.close()
