import os
import torch
import torchvision.models as th_models
import pandas as pd
from collections import OrderedDict
from .wide_resnet import WideResNet
from .densenet import DenseNet3
from .imagenet18 import resnet18
from .reactnet18 import reactnet18
from .reactnet import reactnet
from .ste_wide_resnet import birealresnet8, birealresnet20
from .no_scaling import nsbirealresnet20
from .ste import stebirealresnet20
from .no_shift import noshiftbirealresnet20
from .no_float import nofloatbirealresnet20
from .no_real import norealbirealresnet20
from .no_prelu import noprelubirealresnet20
from .no_scaling_ste import nsstebirealresnet20

def get_model(args):
    relu = torch.nn.ReLU
    if args.model == "rn":
        model = WideResNet(22, args.n_classes, 1, dropRate=args.dropout, ReLU=relu)
    elif args.model == "lrn":
        model = WideResNet(10, args.n_classes, 1, dropRate=args.dropout, ReLU=relu)
    elif args.model == "wrn":
        model = WideResNet(40, args.n_classes, 4, dropRate=args.dropout, ReLU=relu)
    elif args.model == "dn":
        model = DenseNet3(22, args.n_classes, 12, bottleneck=True, dropRate=0.0, ReLU=relu)
    elif args.model == 'resnet18':
        model = resnet18(ReLU=relu)
    elif args.model == 'reactnet18':
        model = reactnet18()
    elif args.model == 'reactnet':
        model = reactnet()
    elif args.model == "brn":
        model = birealresnet20(22, num_classes=args.n_classes)
    elif args.model == "nsstebrn":
        model = nsstebirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "nsbrn":
        model = nsbirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "stebrn":
        model = stebirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "noshiftbrn":
        model = noshiftbirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "noprelubrn":
        model = noprelubirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "nofloatbrn":
        model = nofloatbirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "norealbrn":
        model = norealbirealresnet20(22, num_classes=args.n_classes)
    elif args.model == "blrn":
        model = birealresnet8(10, num_classes=args.n_classes)
    elif args.model == "brn_p2":
        model = birealp2resnet20(22, num_classes=args.n_classes)
    elif args.model == "blrn_p2":
        model = birealp2resnet8(10, num_classes=args.n_classes)
    else:
        raise NotImplementedError

    if args.load_model:
        load_path = os.path.join(args.save_folder,args.load_model)
        state = torch.load(load_path)['model']
        new_state = OrderedDict()
        for k in state:
            # naming convention for data parallel
            if 'module' in k:
                v = state[k]
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = state[k]
        model.load_state_dict(new_state)
        print('Loaded model from {}'.format(args.load_model))
    else:
        #init_weights(model, xavier=True)
        pass

    # Number of model parameters
    args.nparams = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: {}'.format(args.nparams))

    if args.cuda:
        if args.parallel_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    return model


def load_best_model(model, filename):
    if os.path.exists(filename):
        best_model_state = torch.load(filename)['model']
        model.load_state_dict(best_model_state)
        print('Loaded best model from {}'.format(filename))
    else:
        print('Could not find best model')
