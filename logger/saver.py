'''⚡
author: wayn391@mastertones
'''

import os
import time
import yaml
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class Saver(object):
    """Logger & state saver.⚡"""
    def __init__(self, args, initial_global_step=-1):

        self.expdir = args.env.expdir
        self.sample_rate = args.data.sampling_rate
        
        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # Log file
        os.makedirs(self.expdir, exist_ok=True)
        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        # Writer
        os.makedirs(self.expdir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.expdir, 'logs'))
        
        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)


    def log_info(self, msg):
        """Log messages into stdout and file."""
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # display
        print(msg_str)

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_value(self, dict):
        """Log items in TensorBoard."""
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)
    
    def log_audio(self, dict):
        """Log audios in TensorBoard."""
        for k, v in dict.items():
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    def get_interval_time(self):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        self.last_time = cur_time
        return time_interval

    def get_total_time(self):
        total_time = time.time() - self.init_time
        total_time = str(datetime.timedelta(seconds=total_time))[:-5]
        return total_time

    def save_model(self, model, optimizer, postfix: str):
        """Save states."""
        path_pt = os.path.join(self.expdir , f'model_{postfix}.pt')
        print(f' [*] model checkpoint saved: {path_pt}')
        torch.save({'global_step': self.global_step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path_pt)

    def global_step_increment(self):
        self.global_step += 1


