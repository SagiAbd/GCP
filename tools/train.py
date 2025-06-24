# Copyright (c) OpenMMLab. All rights reserved.
""" 
python tools/train.py configs/gcp/mask2former_r50_kazgisa-kostanai.py
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # Add wandb CLI arguments
    parser.add_argument('--wandb-project', type=str, default=None, help='wandb project name')
    parser.add_argument('--wandb-name', type=str, required=True, help='wandb run name')
    parser.add_argument('--wandb-group', type=str, required=True, help='wandb group name')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Compute run_dir and wandb_dir
    run_dir = os.path.join('work_dir', args.wandb_group, args.wandb_name)
    wandb_dir = os.path.join(run_dir, 'wandb')
    cfg.work_dir = run_dir

    # Dynamically inject wandb config if provided
    if args.wandb_project or args.wandb_name or args.wandb_group:
        wandb_backend = {
            'type': 'WandbVisBackend',
            'init_kwargs': {
                'project': args.wandb_project or 'default-project',
                'name': args.wandb_name,
                'group': args.wandb_group,
                'resume': 'allow',
                'allow_val_change': True
            },
            'save_dir': wandb_dir
        }
        cfg.vis_backends = [wandb_backend]
        cfg.visualizer = dict(
            type='TanmlhVisualizer',
            vis_backends=cfg.vis_backends,
            name='visualizer'
        )
        # Optionally add MMDetWandbHook to log_config if present
        if hasattr(cfg, 'log_config') and 'hooks' in cfg.log_config:
            cfg.log_config['hooks'].append(
                dict(
                    type='MMDetWandbHook',
                    init_kwargs=wandb_backend['init_kwargs'],
                    interval=10,
                    log_checkpoint=True,
                    log_checkpoint_metadata=True,
                    num_eval_images=10
                )
            )
            
        
        

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dir',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
    