from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from pathlib import Path



class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg
        self.logger = logging.getLogger('progress')
        self.logger.setLevel(logging.INFO)
        log_format = "%(asctime)s %(message)s"
        date_format = "%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if not cfg.nosave:
            file_handler = logging.FileHandler(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:8f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        if len(metrics_data) > 4:
            metrics_str = ("{:8.4f}, " * 4 + "{:10.2f}").format(*metrics_data[:4], metrics_data[-1])
        else:
            metrics_str = ("{:8.4f}, " * len(metrics_data)).format(*metrics_data)
        self.logger.info(training_str + metrics_str)
        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.cfg.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0
    
    def push(self, metrics):
        self.total_steps += 1
        for k in metrics:
            if k not in self.running_loss:
                self.running_loss[k] = 0.0
            self.running_loss[k] += metrics[k]
        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq - 1:
            self._print_training_status()
            self.running_loss = {}
    
    def write_dict(self, results):
        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)
        for k in results:
            self.writer.add_scalar(k, results[k], self.total_steps)
        self.logger.info(f"Step {self.total_steps}: " + ", ".join(f"{k}: {results[k]}" for k in results))
    
    def close(self):
        if self.writer is not None:
            self.writer.close()
        