import torch
import argparse
import os
import warnings
import shutil
from distutils.dir_util import copy_tree

#paths to change
save_path = '/home/aparen/quantized-scnn/projects/pytorch_fscnn/results'
tb_logs_path = '/home/aparen/quantized-scnn/projects/pytorch_fscnn/tb_logs'
working_path = '/home/aparen/quantized-scnn/projects/pytorch_fscnn/bnew'

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--task', type=str, default='both',
                        help='learning task (sseg, depth, both)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--old_loader', action='store_true', default=False,
                        help='use old loader')
    parser.add_argument('--relu', type=str, default='relu',
                        help='type of rule to use options STE, relu')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')
    # training args
    parser.add_argument('--aux', action='store_true', default=True,
                        help='Auxiliary loss')
    parser.add_argument('--cuda', type=int, default=1,
                          help="use cuda")
    parser.add_argument('--parallel_gpu', dest='parallel_gpu', action='store_true',
                          help="parallel gpu computation")
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='Optimizer to use, choices (adam, sgd, rmsprop)')
    parser.add_argument('--lr', type=float, default=0.045, metavar='LR',
                        help='learning rate (default: 0.045)')
    parser.add_argument('--lr_mode', type=str, default='constant',
                        help='learning rate schedule to use, choices (constant, step, linear, inverse_linear, poly, inverse_poly, cosine)')
    parser.add_argument('--T', type=float, default=[-1], nargs='+',
                        help='lr decay epoch')
    parser.add_argument('--D', type=float, default=[-1], nargs='+',
                        help='lr decay epoch')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       metavar='M', help='w-decay (default: 0.0)')
    parser.add_argument('--scale_reg', action='store_true', default=False,
                          help="scale reg by adam denomination")
    parser.add_argument('--loss', type=str, default='CE',
                        help='which loss function to use')
    # proxquant args
    parser.add_argument('--proxquant', action='store_true', default=False,
                        help='use proxquant')
    parser.add_argument('--float_learn_to_downsample', action='store_true', default=False,
                        help='keep the learn to downsample modular at floating point prescion')
    parser.add_argument('--float_classifier', action='store_true', default=False,
                        help='keep the learn to downsample modular at floating point prescion')
    parser.add_argument('--quant_val', action='store_true', default=False,
                        help='do validation on qunatised model')
    parser.add_argument('--train_acc', action='store_true', default=False,
                        help='calculate train accuacy on training set')
    parser.add_argument('--reg', type=float, default=0.0,
            help='annealing hyperparam (default: 0.0001)')
    parser.add_argument('--freeze_epoch', type=float, default=[-1], nargs='+',
                        help='epoch at which to lock binary parameters to their binary values')
    parser.add_argument('--reg-type', type=str, default='binary',
            help='quantisation type to use')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--teacher', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--load_opt', action='store_true', default=False,
                        help='flag to load optimiser')
    parser.add_argument('--save-folder', default=save_path,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--tb_logs', default=tb_logs_path,
                        help='Directory for tensor board logs')
    parser.add_argument('--name', default=None,
                        help='name of experiement')
    parser.add_argument('--tag', default=None,
                        help='tag')
    parser.add_argument('--no_log', dest='log', action='store_false', default=True,
                        help='store text log')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='set debug mode')
    parser.add_argument('--visdom', dest='visdom', action='store_true', default=False,
                          help='use visdom')
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='use tensorboard')

    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--eval-binary', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # the parser
    args = parser.parse_args()
    args.freeze_epoch = list(args.freeze_epoch)
    args.T = list(args.T)
    args.D = list(args.D)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.load_model = args.resume
    set_xp_name(args)
    set_num_classes(args)
    if 'mirror' in args.reg_type or 'ste' in args.reg_type:
        args.quant_val = True
        args.train_acc = True
    return args

def set_num_classes(args):
    if args.dataset == 'cifar10':
        args.n_classes = 10
        args.train_size = 45000
        args.input_dims = 3*32**2
    elif args.dataset == 'cifar100':
        args.n_classes = 100
        args.train_size = 45000
        args.input_dims = 3*32**2
    elif args.dataset == 'mnist':
        args.n_classes = 10
        args.train_size = 45000
        args.input_dims = 28**2
    elif 'svhn' in args.dataset:
        args.n_classes = 10
        args.train_size = 598388
        args.input_dims = 3*32**2
    elif args.dataset == 'imagenet':
        args.n_classes = 1000
        args.train_size = 1231166
        args.input_dims = 3*224**2
    elif args.dataset == 'tiny_imagenet':
        args.n_classes = 200
        args.train_size = 100000
        args.input_dims = 3*64**2
    else:
        raise ValueError
    args.n_batches_per_epoch = args.train_size//args.batch_size + int(not(args.train_size%args.batch_size==0))

def set_xp_name(args):
    if 'our' in args.reg_type:
        args.reg_type = 'bnew_binary'

    if args.debug:
        args.batch_size = 8
        args.epochs = 2

    if args.dataset == 'toy':
        args.model = "matrix_fac_10"
        args.batch_size = 100
        args.epochs = 100

    if args.name is None:
        xp_name = '{data}/'.format(data=args.dataset)
        xp_name += "{model}{data}-{opt}--lr-{lr_mode}-{lr}--wd-{wd}--b-{b}-epoch-{epoch}-reg-{reg}-type-{reg_type}-{tag}".format(model=args.model, data=args.dataset, opt=args.opt, lr=args.lr, lr_mode=args.lr_mode, wd=args.weight_decay, b=args.batch_size, epoch=args.epochs, reg=args.reg, reg_type=args.reg_type, tag=args.tag)
        args.xp_name_no_dir = xp_name

        args.xp_name = os.path.join(args.save_folder,xp_name)
        if args.debug:
            args.xp_name += "--debug"

    if not args.debug:
        if os.path.exists(args.xp_name):
            if not args.debug:
                print('An experiment already exists at {}'.format(os.path.abspath(args.xp_name)))
                a = input('delete directory ? (type yes): ')
                if not(( a == 'yes') or (a == 'Yes')):
                    warnings.warn('An experiment already exists at {}'.format(os.path.abspath(args.xp_name)))
                    raise RuntimeError
        else:
            os.makedirs(args.xp_name)
            print(args.xp_name)
            copy_tree(working_path, os.path.join(args.xp_name+'/code/'))
