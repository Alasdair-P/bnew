import torch

def get_optimizer(args, parameters, other_parameters, weight_parameters):
    parameters = list(parameters)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=True)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam_wd":
        optimizer = torch.optim.Adam(
		    [{'params' : other_parameters},
		    {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
		    lr=args.lr,)
    elif args.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=0.9, eps=1.0)
    else:
        raise ValueError(args.opt)

    if args.load_opt:
        args.load_opt = args.load_model
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))
    return optimizer
