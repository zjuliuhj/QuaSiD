import os
import torch
import argparse
import torch.distributed as dist
from engine.logger import Logger
from engine.solver import Trainer
from data.build_dataloader import build_dataloader, build_dataloader_conditional, build_dataloader_forecasting
from evaluation import build_evaluator
from utils.runtime import (
    instantiate_from_config,
    load_yaml_config,
    materialize_runtime_config,
    merge_opts_to_config,
    seed_everything,
    validate_config,
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def setup_distributed(args):
    """Initialize single-process or torchrun DDP device state on the parsed args."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1 and args.gpu is not None:
        raise ValueError("Do not pass --gpu when launching DDP. Use torchrun and let LOCAL_RANK choose the device.")

    args.distributed = env_world_size > 1
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.is_main_process = True

    if not args.distributed:
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        return

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices.")

    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.is_main_process = args.rank == 0
    args.device = f"cuda:{args.local_rank}"


def cleanup_distributed(args):
    """Tear down the active distributed process group after train/eval completion."""
    if getattr(args, "distributed", False) and dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(getattr(args, "local_rank", 0))])
        else:
            dist.barrier()
        dist.destroy_process_group()

def build_parser():
    """Build the public CLI for QuaSiD train and checkpoint evaluation."""
    parser = argparse.ArgumentParser(description='QuaSiD V5 training and checkpoint evaluation')
    parser.add_argument('--name', type=str, required=True, help='experiment name')
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help='Experiment name used to load checkpoints during evaluation. Defaults to --name.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='path of config file')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # args for randomness
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--mode', type=str, default='uncond', choices=['uncond', 'cond', 'forecast'],
                        help='Generation mode: uncond, cond, or forecast.')
    parser.add_argument('--milestone', type=int, default=5, help='Checkpoint number')
    parser.add_argument('--eval_tag', type=str, default=None,
                        help='Optional single-directory label below eval/, e.g. milestone-3.')

    # args for config overrides
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--model_name', type=str, default='quasid_sde',
                        help='The name of the model, used for path naming')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name used for path grouping, e.g. etth1 in quasid_sde/etth1/etth1_seq128.')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='How many times to run the sampling process')
    parser.add_argument('--visualization_compare', type=int, default=3000,
                        help='Number of real/synthetic samples used in PCA/t-SNE/KDE visualizations.')
    parser.add_argument('--eval_sample_limit', type=int, default=None,
                        help='Maximum number of generated samples during evaluation. Defaults to the full dataset.')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip metric computation during checkpoint evaluation.')
    parser.add_argument('--metric_samples', type=int, default=3000,
                        help='Maximum number of paired real/synthetic samples used for metrics.')
    parser.add_argument('--discriminative_iterations', type=int, default=2000,
                        help='Training steps for the post-hoc discriminative metric.')
    parser.add_argument('--predictive_iterations', type=int, default=5000,
                        help='Training steps for the post-hoc predictive metric.')
    parser.add_argument('--ensemble_chunk_size', type=int, default=4,
                        help='Forecasting only: number of sampled future paths generated together per condition batch.')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip evaluation visualization generation.')
    return parser


def parse_args(argv=None):
    """Parse CLI arguments and derive experiment paths without filesystem side effects."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    if args.opts and len(args.opts) % 2 != 0:
        parser.error("Each config override must be a name/value pair.")

    removed_flags = {"--output", "--sample", "--cudnn_deterministic"}
    for opt in args.opts:
        if opt in removed_flags:
            parser.error(f"{opt} was removed from the QuaSiD CLI.")

    if args.dataset_name is None:
        args.dataset_name = os.path.splitext(os.path.basename(args.config_file))[0]

    args.ckpt_name = args.name if args.train or args.ckpt_name is None else args.ckpt_name
    run_name = args.name if args.train else args.ckpt_name
    if args.eval_tag is not None and (
        args.eval_tag in {".", ".."} or os.path.basename(args.eval_tag) != args.eval_tag
    ):
        parser.error("--eval_tag must be a single directory name.")
    args.experiment_dir = os.path.join('experiments', args.mode, args.model_name, args.dataset_name, run_name)
    args.ckpt_folder = os.path.join(args.experiment_dir, 'ckpt')
    args.save_dir = os.path.join(args.experiment_dir, 'log')
    args.eval_dir = os.path.join(
        args.experiment_dir,
        'eval',
        args.eval_tag,
    ) if args.eval_tag is not None else os.path.join(args.experiment_dir, 'eval')
    return args


def create_output_dirs(args):
    """Create experiment directories after argument parsing succeeds."""
    os.makedirs(args.ckpt_folder, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

def attach_runtime_device(runtime_config, device):
    """Bind the resolved torch device to model params before constructor-side placement."""
    runtime_config['model']['params']['device'] = str(device)
    return runtime_config

def main():
    args = parse_args()
    create_output_dirs(args)
    setup_distributed(args)

    seed_everything(args.seed)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    validate_config(config, args.mode)

    if args.distributed and not args.train:
        if args.is_main_process:
            print("DDP checkpoint evaluation is disabled. Run evaluation with a single process.")
        cleanup_distributed(args)
        return

    logger = Logger(args) if args.is_main_process else None
    if logger is not None:
        logger.save_config(config)

    runtime_config = materialize_runtime_config(config, args.mode)
    runtime_config = attach_runtime_device(runtime_config, args.device)
    model = instantiate_from_config(runtime_config['model']).to(torch.device(args.device))

    if args.mode == 'uncond':
        dataloader_info = build_dataloader(runtime_config, args)
        test_dataloader_info = None
    elif args.mode == 'cond':
        dataloader_info = build_dataloader_conditional(runtime_config, args, period='train')
        test_dataloader_info = build_dataloader_conditional(runtime_config, args, period='test')
    elif args.mode == 'forecast':
        dataloader_info = build_dataloader_forecasting(runtime_config, args, period='train')
        test_dataloader_info = build_dataloader_forecasting(runtime_config, args, period='test')

    trainer = Trainer(config=runtime_config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
        if logger is not None:
            logger.close()
        cleanup_distributed(args)
        return

    trainer.load(args.milestone)
    evaluator = build_evaluator(
        args=args,
        config=config,
        trainer=trainer,
        dataloader_info=dataloader_info,
        test_dataloader_info=test_dataloader_info,
        logger=logger,
    )
    evaluator.evaluate()

    if logger is not None:
        logger.close()
    cleanup_distributed(args)

if __name__ == '__main__':
    main()
