import os
import numpy as np

from tensorboardX import SummaryWriter

class TensorBoard:
    def __init__(self, logdir):
        self.writer = SummaryWriter(logdir)
        self.epoch = 0
        self.obj_count = 0

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step=global_step)

    def update_epoch(self):
        self.epoch += 1

    def plot_for_current_epoch(self, tag, value):
        self.log_scalar(tag, value, self.epoch)

    def plot_obj_val(self, value):
        self.obj_count += 1
        self.log_scalar('Training Objective Value', value, self.obj_count)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
