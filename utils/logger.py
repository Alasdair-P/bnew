import os
import sys
import torch
import mlogger
import random
import numpy as np
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('tensor board not found')

def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        visdom_plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        visdom_plotter = None

    if args.tensorboard:
        print('args.tensorboard:', args.tensorboard)
        summary_writer = SummaryWriter(log_dir=args.tb_dir)
    else:
        summary_writer = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(visdom_plotter=visdom_plotter, summary_writer=summary_writer)

    vars_ = copy.deepcopy(vars(args))
    for key, value in vars_.items():
        if value is None or type(value) is list:
            vars_[key] = str(value)
    print(vars_)

    xp.config.update(**vars_)

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.obj = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Weight-Norm")
    xp.train.grad_norm = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Grad-Norm")

    xp.train.step_size = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="validation")
    xp.val.iou = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="validation")
    xp.val.pic_acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='validation')
    xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter,  summary_writer=summary_writer, plot_title="Time", plot_legend='test')

    if args.visdom:
        visdom_plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        visdom_plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')
        xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

    return xp


