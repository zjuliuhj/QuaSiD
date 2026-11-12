from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from utils.runtime import save_config_to_yaml, write_args


class Logger(object):
    def __init__(self, args, solver_config=None):
        self.args = args
        self.save_dir = args.save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
            
        # save the args and config
        self.config_dir = os.path.join(self.save_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        file_name = os.path.join(self.config_dir, 'args.txt')
        write_args(args, file_name)

        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
        solver_config = solver_config or {}
        self._tensorboard_enabled = bool(args.tensorboard)
        self.max_queue = int(solver_config.get('tensorboard_max_queue', 1000))
        self.flush_secs = float(solver_config.get('tensorboard_flush_secs', 120))
        if self.max_queue < 1 or self.flush_secs <= 0:
            raise ValueError('TensorBoard queue size and flush interval must be positive.')
        self.tb_writer = None
        self.log_dir = log_dir
        self._closed = False
        if self.enabled:
            self.log_info('using tensorboard')

    @property
    def enabled(self):
        return self._tensorboard_enabled and not self._closed

    def _writer(self):
        """Avoid empty event files when an evaluation does not log metrics."""
        if self.tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(
                log_dir=self.log_dir, max_queue=self.max_queue, flush_secs=self.flush_secs,
            )
        return self.tb_writer

    def log_scalars(self, scalars, global_step):
        """Pack related CPU scalars into one event, preserving their existing tags."""
        if self.enabled and scalars:
            from tensorboard.compat.proto.summary_pb2 import Summary

            summary = Summary(value=[
                Summary.Value(tag=tag, simple_value=float(value))
                for tag, value in scalars.items()
            ])
            self._writer().file_writer.add_summary(summary, global_step)
            
    def save_config(self, config):
        save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        print(info)
        info = str(info)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        info = '{}: {}'.format(time_str, info)
        if not info.endswith('\n'):
            info += '\n'
        self.text_writer.write(info)
        self.text_writer.flush()

    def add_scalar(self, **kargs):
        """Log a scalar variable."""
        if self.enabled:
            self._writer().add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.enabled:
            self._writer().add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.enabled:
            self._writer().add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.enabled:
            self._writer().add_images(**kargs)

    def flush(self):
        """Flush text and TensorBoard event writers to disk."""
        self.text_writer.flush()
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self):
        if self._closed:
            return
        self.flush()
        self.text_writer.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        self._closed = True
