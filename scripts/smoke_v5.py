"""Exercise the complete V5 train/checkpoint/evaluate path on CPU in a temporary directory."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    base = yaml.safe_load((ROOT / 'config/unconditional/sines.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='quasid-v5-smoke-') as tmp:
        work = Path(tmp)
        for wrapper in ('cont', 'disc'):
            config = copy.deepcopy(base)
            config['model']['target'] = f'models.quasid.quasid_{wrapper}.QuaSiD_{wrapper}'
            config['model']['params'].update(
                seq_length=16, channels=2, emb_size=32, corr_rank=3, num_layers=1,
                max_step=8, sampling_steps=2, backbone='dit_solver_v5', attn_backend='sdpa')
            config['solver'].update(train_steps=2, save_cycle=2, ckpt_count=1,
                                    grad_accum_every=1, mixed_precision='no')
            config['solver']['ema'].update(update_interval=1)
            config['solver']['scheduler'] = {
                'target': 'engine.lr_sch.CosineAnnealingLRWithWarmup',
                'params': {'T_max': 2, 'warmup': 0, 'min_lr': 0.0001,
                           'warmup_lr': 0.0001, 'verbose': False}}
            config['dataloader']['dataset']['params'].update(window=16, num=8, dim=2)
            config['dataloader'].update(batch_size=2, sample_size=2, num_workers=0,
                                        pin_memory=False, persistent_workers=False)
            config_path = work / f'{wrapper}.yaml'
            config_path.write_text(yaml.safe_dump(config))
            common = [sys.executable, str(ROOT / 'main.py'), '--name', 'smoke_v5',
                      '--mode', 'uncond', '--model_name', f'quasid_{wrapper}',
                      '--dataset_name', 'sines', '--config_file', str(config_path)]
            for flags in (['--train'], ['--milestone', '1', '--eval_sample_limit', '4',
                                      '--skip_metrics', '--skip_visualization']):
                result = subprocess.run(common + flags, cwd=work, env=environment,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(result.stdout[-15000:])
            checkpoint = work / f'experiments/uncond/quasid_{wrapper}/sines/smoke_v5/ckpt/ckpt-1.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert state['step'] == 2
            assert all(torch.isfinite(value).all() for value in state['model'].values())
            print(f'PASS: V5 {wrapper.upper()} training, finite checkpoint, EMA reload, and sampling')
    print('PASS: complete CPU smoke check; temporary artifacts removed')


if __name__ == '__main__':
    main()
