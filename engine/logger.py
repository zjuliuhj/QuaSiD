from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import shutil
import tempfile
import time
from utils.runtime import save_config_to_yaml, write_args


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # save the args and config
        self.config_dir = os.path.join(self.save_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        file_name = os.path.join(self.config_dir, 'args.txt')
        write_args(args, file_name)

        self.log_dir = Path(self.save_dir) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_writer = (self.log_dir / 'log.txt').open('a')
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            # Network filesystem writes can fill SummaryWriter's queue and stall rank 0.
            tb_root = Path(os.environ.get(
                'QUASID_TENSORBOARD_ROOT', Path(tempfile.gettempdir()) / 'quasid-tensorboard'
            ))
            tb_root.mkdir(parents=True, exist_ok=True)
            run_name = Path(self.save_dir).parent.name
            self.tb_dir = Path(tempfile.mkdtemp(prefix=f'{run_name}-', dir=tb_root))
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir), flush_secs=30)
            self.log_info(f'TensorBoard live events: {self.tb_dir}')
        else:
            self.tb_writer = None

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
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_images(**kargs)

    def flush(self):
        """Flush text and TensorBoard event writers to disk."""
        self.text_writer.flush()
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self):
        self.flush()
        self.text_writer.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
            # Archive only after the writer has stopped; retain local events for recovery.
            for event_file in self.tb_dir.glob('events.out.tfevents.*'):
                shutil.copy2(event_file, self.log_dir / event_file.name)
