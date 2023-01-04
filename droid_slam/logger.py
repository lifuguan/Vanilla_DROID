
import torch
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 50

class Logger:
    def __init__(self, args, scheduler):
        self.current_steps = 0
        self.total_steps = args.steps
        self.name = args.name
        self.running_loss = {}
        self.writer = None
        self.scheduler = scheduler

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter('runs/%s' % self.name)
            print([k for k in self.running_loss])

        lr = self.scheduler.get_lr().pop()
        metrics_data = {k:self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()}
        training_str = "[STEPS {:6d} / {:6d}, lR : {:10.7f}] ".format(self.current_steps+1, self.total_steps, lr)
        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str, metrics_data)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.current_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.current_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.current_steps += 1

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.current_steps)

    def close(self):
        self.writer.close()

