import os
import torch

def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def save_checkpoint(model, optimizer, args, is_best=False, is_quantised=False):
    if is_quantised:
        if is_best:
            save_state(model, optimizer, '{}/best_quant_model.pkl'.format(args.xp_name))
        else: 
            save_state(model, optimizer, '{}/quant_model.pkl'.format(args.xp_name))
    else:
        if is_best:
            save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name))
        else: 
            save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name))
