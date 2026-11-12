import os
import sys
import copy
import time
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler
from utils.ema import EMA
from utils.runtime import get_model_parameters_info, instantiate_from_config



sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl, sampler=None):
    """Yield batches forever, resetting a distributed sampler at each epoch."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for data in dl:
            yield data
        epoch += 1


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        """Prepare device placement, DDP wrapping, optimizer, EMA, scheduler, and runtime bookkeeping."""
        super().__init__()
        self.args, self.config = args, config
        self.distributed = bool(getattr(args, "distributed", False))
        self.rank = int(getattr(args, "rank", 0))
        self.local_rank = int(getattr(args, "local_rank", 0))
        self.world_size = int(getattr(args, "world_size", 1))
        self.is_main_process = bool(getattr(args, "is_main_process", True))

        if self.distributed and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif args.gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            self.device = next(model.parameters()).device

        model = model.to(self.device)
        if self.distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = model
        self.train_model = self.model.module if self.distributed else self.model
        self.train_num_steps = config['solver']['train_steps']
        self.grad_accum_every = config['solver']['grad_accum_every']
        self.log_interval = int(config['solver'].get('log_interval', 50))
        if self.log_interval < 1:
            raise ValueError('solver.log_interval must be positive.')
        self.ckpt_count = int(config['solver'].get('ckpt_count', 5))
        self.save_cycle = int(config['solver'].get(
            'save_cycle',
            max(1, self.train_num_steps // self.ckpt_count),
        ))
        self.dl = cycle(dataloader['dataloader'], sampler=dataloader.get('sampler'))
        self.step = 0
        self.milestone = 0
        self.logger = logger
        self.ckpt_folder = Path(args.ckpt_folder)
        batch_info = dataloader.get('batch_info', {})
        self.global_batch_size = int(batch_info.get(
            "global_batch_size", config["dataloader"]["batch_size"]
        ))
        self.samples_per_optimizer_step = self.global_batch_size * self.grad_accum_every
        # Persist this value in checkpoints so elapsed-time curves continue
        # monotonically after a resumed training run.
        self.elapsed_train_seconds = 0.0

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.train_model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = copy.deepcopy(config['solver']['scheduler'])
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)
        # Unknown schedulers keep the previous metric-driven behavior.
        self.scheduler_requires_metric = bool(getattr(self.sch, 'requires_metric', True))

        self.mixed_precision = str(config['solver'].get('mixed_precision', 'no')).lower()
        self.use_amp = self.device.type == "cuda" and self.mixed_precision in {"fp16", "bf16"}
        self.amp_dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.mixed_precision == "fp16")

        if self.logger is not None:
            trainable_parameter_count = sum(
                parameter.numel() for parameter in self.train_model.parameters() if parameter.requires_grad
            )
            total_parameter_count = sum(parameter.numel() for parameter in self.train_model.parameters())
            if self.logger.enabled and getattr(args, 'train', True):
                self.logger.log_scalars({
                    'model/trainable_parameters': trainable_parameter_count,
                    'model/total_parameters': total_parameter_count,
                }, self.step)
            self.logger.log_info(str(get_model_parameters_info(self.train_model)))
            self.logger.log_info(
                "Batch size. global: {} | local_per_gpu: {} | world_size: {}".format(
                    batch_info.get("global_batch_size", config["dataloader"]["batch_size"]),
                    batch_info.get("local_batch_size", config["dataloader"]["batch_size"]),
                    batch_info.get("world_size", self.world_size),
                )
            )
            self.logger.log_info(
                "Mixed precision: {}".format(self.mixed_precision if self.use_amp else "disabled")
            )

    def _barrier(self):
        """Synchronize all DDP ranks when distributed training is active."""
        if self.distributed:
            dist.barrier(device_ids=[self.local_rank])

    def _mean_across_ranks(self, value):
        """Reduce detached metrics on device without a CPU/GPU round trip."""
        tensor = value.detach().clone()
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor

    def _synchronize_device(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _conditional_inst_stats(self, x_raw, condition):
        """Return per-sample stats from a sequence condition or conditional target."""
        if not getattr(self.ema.ema_model, "inst_norm", False):
            return None

        source = condition if condition is not None and condition.ndim == 3 else x_raw
        if source is None:
            raise ValueError("Conditional instance normalization requires a BLC statistics source.")
        _, stats = self.ema.ema_model.inst_normalize(source)
        return stats

    def _move_batch_to_device(self, data):
        """Move a tensor batch or a sequence of tensor batches onto the trainer device."""
        if isinstance(data, (list, tuple)):
            return [item.to(self.device, non_blocking=True) for item in data]
        return data.to(self.device, non_blocking=True)

    def save(self, milestone, verbose=False):
        """Save model, EMA, optimizer state, and current training step on the main process."""
        if not self.is_main_process:
            return
        ckpt_path = self.ckpt_folder / f'ckpt-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(ckpt_path)))
        data = {
            'step': self.step,
            'model': self.train_model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'elapsed_train_seconds': self.elapsed_train_seconds,
        }
        torch.save(data, str(ckpt_path))

    def load(self, milestone, verbose=False):
        """Load a checkpoint milestone into the train model, optimizer, EMA, and step counters."""
        ckpt_path = self.ckpt_folder / f'ckpt-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(ckpt_path)))
        data = torch.load(str(ckpt_path), map_location=self.device)
        self.train_model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone
        self.elapsed_train_seconds = float(data.get('elapsed_train_seconds', 0.0))

    def train(self):
        """Run the main optimization loop with gradient accumulation, AMP, EMA, logging, and checkpoints."""
        data_iter = iter(self.dl)
        self.model.train()

        show_progress = self.is_main_process and sys.stderr.isatty()
        track_metrics = show_progress or (self.logger is not None and self.logger.enabled)
        if self.distributed:
            # All ranks must make the same decision about metric collectives.
            flag = torch.tensor(int(track_metrics), device=self.device)
            dist.broadcast(flag, src=0)
            track_metrics = bool(flag.item())
        window_metrics = torch.zeros(2, dtype=torch.float64, device=self.device) if track_metrics else None
        window_steps = 0
        self._synchronize_device()
        initial_elapsed = self.elapsed_train_seconds
        training_start = last_report_time = time.perf_counter()

        progress = tqdm(
            initial=self.step, total=self.train_num_steps, desc='Training',
            disable=not show_progress, mininterval=1.0,
        )
        with progress as pbar:
            while self.step < self.train_num_steps:
                total_loss_step = None
                for _ in range(self.grad_accum_every):
                    data = self._move_batch_to_device(next(data_iter))
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        loss = self.model(data)
                    loss = loss / self.grad_accum_every
                    self.scaler.scale(loss).backward()
                    if track_metrics or self.scheduler_requires_metric:
                        # Accumulate detached values on device. Double precision
                        # preserves the previous sum of Python loss scalars.
                        value = loss.detach().double()
                        total_loss_step = value if total_loss_step is None else total_loss_step + value

                scheduler_metric = None
                if self.scheduler_requires_metric:
                    # Plateau schedules genuinely require a global loss each step.
                    scheduler_metric = self._mean_across_ranks(total_loss_step).item()

                self.scaler.unscale_(self.opt)
                grad_norm = clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                if self.scheduler_requires_metric:
                    self.sch.step(scheduler_metric)
                else:
                    self.sch.step()
                self.opt.zero_grad()
                self.ema.update()
                self.step += 1

                if track_metrics:
                    window_metrics[0].add_(total_loss_step)
                    window_metrics[1].add_(grad_norm.detach())
                    window_steps += 1

                if self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self._synchronize_device()
                    self.elapsed_train_seconds = initial_elapsed + time.perf_counter() - training_start
                    self.save(self.milestone)

                if track_metrics and (self.step % self.log_interval == 0 or self.step == self.train_num_steps):
                    metrics = self._mean_across_ranks(window_metrics) / window_steps
                    self._synchronize_device()
                    if self.is_main_process:
                        # One transfer per reporting window; nothing is fetched
                        # for TensorBoard when it and terminal progress are off.
                        mean_loss, mean_grad_norm = metrics.cpu().tolist()
                        now = time.perf_counter()
                        window_seconds = max(now - last_report_time, 1e-12)
                        self.elapsed_train_seconds = initial_elapsed + now - training_start
                        if self.logger is not None and self.logger.enabled:
                            self.logger.log_scalars({
                                'train/loss': mean_loss,
                                'train/lr': self.opt.param_groups[0]['lr'],
                                'train/grad_norm': mean_grad_norm,
                                'train/step_time_s': window_seconds / window_steps,
                                'train/elapsed_time_s': self.elapsed_train_seconds,
                                'train/throughput_samples_per_s': self.samples_per_optimizer_step * window_steps / window_seconds,
                                'train/samples_seen': self.step * self.samples_per_optimizer_step,
                            }, self.step)
                        pbar.set_description(f'loss: {mean_loss:.6f}', refresh=False)
                        # The next window includes this report's submission time.
                        last_report_time = now
                    window_metrics.zero_()
                    window_steps = 0
                pbar.update(1)

        self._synchronize_device()
        self._barrier()
        self.elapsed_train_seconds = initial_elapsed + time.perf_counter() - training_start
        if self.is_main_process:
            print('training complete')
        if self.logger is not None:
            self.logger.log_info(f'Training Done. Total Steps: {self.step}')

    def sample(self, num, size_every):
        """Generate unconditional samples in chunks with the EMA model."""
        if self.logger is not None:
            self.logger.log_info('Begin to sample...')

        all_samples = []
        current_num = 0

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            while current_num < num:
                batch_size = min(size_every, num - current_num)
                sample = self.ema.ema_model.generate_unconditional(batch_size=batch_size)

                all_samples.append(sample.detach().cpu().numpy())
                current_num += batch_size

        samples = np.concatenate(all_samples, axis=0)

        if self.logger is not None:
            self.logger.log_info('Sampling done.')
        return samples

    def conditional_sample(self, num, raw_dataloader, sampling_steps=50):
        """Generate one conditional sample per condition and return generated samples plus available targets."""
        all_samples, all_reals = [], []
        self.ema.ema_model.eval()

        current_count = 0

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            for batch in tqdm(raw_dataloader, desc="Conditional Sampling"):
                if current_count >= num:
                    break

                if isinstance(batch, (list, tuple)):
                    x_raw = batch[0].to(self.device, non_blocking=True).float()
                    cond = batch[1].to(self.device, non_blocking=True).float()
                else:
                    cond = batch.to(self.device, non_blocking=True).float()
                    x_raw = None

                actual_batch_size = cond.shape[0]
                inst_stats = self._conditional_inst_stats(x_raw, cond)

                sample = self.ema.ema_model.generate_conditional(
                    batch_size=actual_batch_size,
                    cond=cond,
                    sampling_steps=sampling_steps,
                    inst_stats=inst_stats,
                )

                all_samples.append(sample.detach().cpu().numpy())
                if x_raw is not None:
                    all_reals.append(x_raw.detach().cpu().numpy())

                current_count += actual_batch_size

        samples = np.concatenate(all_samples, axis=0)[:num]
        reals = np.concatenate(all_reals, axis=0)[:num] if all_reals else np.empty((0,))

        if self.logger is not None:
            self.logger.log_info('Conditional sampling done.')

        return samples, reals

    def conditional_ensemble_sample(
        self,
        num,
        raw_dataloader,
        ensemble_size,
        ensemble_chunk_size=4,
        sampling_steps=50,
    ):
        """Generate multiple conditional paths per condition, chunking ensemble members to control memory."""
        all_samples, all_reals = [], []
        self.ema.ema_model.eval()

        current_count = 0
        ensemble_size = int(ensemble_size)
        ensemble_chunk_size = int(ensemble_chunk_size)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            for batch in tqdm(raw_dataloader, desc="Conditional Ensemble Sampling"):
                if current_count >= num:
                    break

                if isinstance(batch, (list, tuple)):
                    x_raw = batch[0].to(self.device, non_blocking=True).float()
                    cond = batch[1].to(self.device, non_blocking=True).float()
                else:
                    cond = batch.to(self.device, non_blocking=True).float()
                    x_raw = None

                batch_size = min(cond.shape[0], num - current_count)
                cond = cond[:batch_size]
                x_raw = x_raw[:batch_size] if x_raw is not None else None
                base_stats = self._conditional_inst_stats(x_raw, cond)
                batch_runs = []

                for start in range(0, ensemble_size, ensemble_chunk_size):
                    chunk_size = min(ensemble_chunk_size, ensemble_size - start)
                    # Repeat each condition for a small ensemble chunk, then reshape samples back to
                    # (chunk, batch, length, channels) before concatenating across chunks.
                    cond_chunk = cond.repeat_interleave(chunk_size, dim=0)
                    inst_stats = None
                    if base_stats is not None:
                        inst_stats = tuple(
                            stat.repeat_interleave(chunk_size, dim=0) for stat in base_stats
                        )

                    sample = self.ema.ema_model.generate_conditional(
                        batch_size=cond_chunk.shape[0],
                        cond=cond_chunk,
                        sampling_steps=sampling_steps,
                        inst_stats=inst_stats,
                        )
                    sample = sample.reshape(batch_size, chunk_size, *sample.shape[1:]).permute(1, 0, 2, 3)
                    batch_runs.append(sample.detach())

                all_samples.append(torch.cat(batch_runs, dim=0).cpu().numpy())
                if x_raw is not None:
                    all_reals.append(x_raw.detach().cpu().numpy())
                current_count += batch_size

        samples = np.concatenate(all_samples, axis=1)[:, :num]
        reals = np.concatenate(all_reals, axis=0)[:num] if all_reals else np.empty((0,))

        if self.logger is not None:
            self.logger.log_info('Conditional ensemble sampling done.')

        return samples, reals
