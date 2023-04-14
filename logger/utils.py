"""Utilities"""

import os
import yaml
import torch


def traverse_dir(root_dir, extension, is_pure=False, is_ext=True):
    """glob, then format the path, finally sort if needed.
    Args:
        is_ext - Whether to include extention string as a file path
        is_pure - Whether to return relative path toward `root_dir` or absolute path
    """
    file_list = []
    # current_dir, child_dirs, child_files
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path - full path | relative to `root_dir`
                full_path = os.path.join(root, file)
                pure_path = full_path[len(root_dir)+1:] if is_pure else full_path

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]

                file_list.append(pure_path)
    return file_list


class DotDict(dict):
    """easy-access object.⚡"""
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_config(path_config):
    """Load yaml file as easy-access config.⚡"""
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args


def load_model(expdir, model, optimizer, device='cpu'):
    """Load most fresh model states in a directory.⚡"""
    path = os.path.join(expdir, 'model_')
    path_pts = traverse_dir(expdir, '.pt', is_ext=False)
    global_step = 0
    if len(path_pts) > 0:
        steps = [s[len(path):] for s in path_pts]
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        # Biggest step || best.pt
        if maxstep > 0:
            path_pt = path+str(maxstep)+'.pt'
        else:
            path_pt = path+'best.pt'
        print(' [*] restoring model from', path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt['global_step']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    return global_step, model, optimizer
