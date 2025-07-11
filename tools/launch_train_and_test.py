import os
import subprocess
from vizualise.comparator import run_comparison_from_config
import sys
sys.path.insert(0, 'D:\Sagi\GCP\GCP')
"""
$env:PYTHONPATH = "D:\Sagi\GCP\GCP"
python tools/launch_train_and_test.py
"""

CONFIG = {
    "wandb_group": "mask2former_swin-l_training",
    "wandb_name": "mask2former_swin-l_e50_lre-4-cosine_whu-mix_v2",
    "wandb_project": "building-segmentation-gcp",
    "train_config": "configs/gcp/mask2former_swinl_whu-mix-vector.py",
    # "load_from": "checkpoints/mask2former_r50_pretrained_50e_whu-mix-vector.pth",
    # "resume_from": '/kaggle/input/mask2former_swinl/pytorch/default/2/epoch_4.pth',
    # "resume": False,  # Automatically resume from last available epoch if no resume_from/load_from specified
    "test1_config": "configs/gcp/mask2former_swinl_whu-mix-vector.py",
    # "test1_checkpoint": "D:\Sagi\GCP\GCP\checkpoints\gcp_e5_lre-4_kostanai-afs_v1.pth",
    "test2_config": "configs/gcp/gcp_r50_kazgisa-kostanai.py",
    "test2_checkpoint": "checkpoints/gcp_r50_pretrained_12e_whu-mix-vector.pth",
    "gpus": 2,
    "comparison_limit": 30,
    "show_image_width": 10,
    "max_epochs": 50,
    "visualize_val_each_epoch": False  # New option
}

def run_command(cmd):
    print(f"\n[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    # Always resolve GCP root directory (the directory containing this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gcp_root = script_dir  # This script is inside GCP/tools/, so GCP root is one level up
    if os.path.basename(gcp_root) == 'tools':
        gcp_root = os.path.dirname(gcp_root)
    output_root = gcp_root
    run_dir = os.path.join(output_root, 'work_dir', CONFIG["wandb_group"], CONFIG["wandb_name"])
    os.makedirs(run_dir, exist_ok=True)
    comparison_dir = os.path.join(run_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Train
    train_cmd = [
        'python', '-m', 'torch.distributed.launch', '--nproc_per_node=2',
        'tools/train.py',
        CONFIG["train_config"],
        '--wandb-group', CONFIG["wandb_group"],
        '--wandb-name', CONFIG["wandb_name"],
        '--wandb-project', CONFIG["wandb_project"],
        '--launcher', 'pytorch',
        '--work-dir', run_dir
    ]
    if CONFIG.get("resume_from"):
        train_cmd += ['--resume', CONFIG["resume_from"]]
    elif CONFIG.get("load_from"):
        train_cmd += ['--cfg-options', f'load_from={CONFIG["load_from"]}']
    elif CONFIG.get("resume"):
         train_cmd += ['--resume']
    # Add visualization interval override if requested
    if CONFIG.get("visualize_val_each_epoch", False):
        train_cmd += ['--visualize-val-each-epoch']
    else:
        train_cmd += ['--cfg-options', 'default_hooks.visualization.draw=False']
    run_command(train_cmd)

    # 2. Dynamically set test1_checkpoint to the last epoch checkpoint
    # max_epochs = CONFIG.get("max_epochs", 10)
    # checkpoint_name = f"epoch_{max_epochs}.pth"
    # checkpoint_path = os.path.join(run_dir, checkpoint_name)
    # CONFIG["test1_checkpoint"] = checkpoint_path

    # # 3. Test 1 (output dir: <wandb_name>_test)
    # test1_out_dir = os.path.join(run_dir, 'test1')
    # os.makedirs(test1_out_dir, exist_ok=True)
    # test1_vis_dir = os.path.join(test1_out_dir, 'visualizations')
    # os.makedirs(test1_vis_dir, exist_ok=True)
    # test1_metrics = os.path.join(test1_out_dir, 'metrics.pkl')
    # test1_cmd = [
    #     'python', 'tools/test.py',
    #     CONFIG["test1_config"],
    #     CONFIG["test1_checkpoint"],
    #     '--work-dir', test1_out_dir,
    #     '--show-dir', "visualizations",
    #     '--out', test1_metrics
    # ]
    # # run_command(test1_cmd + ['--launcher', 'none'])

    # # 4. Test 2
    # test2_out_dir = os.path.join(run_dir, 'test2')
    # os.makedirs(test2_out_dir, exist_ok=True)
    # test2_vis_dir = os.path.join(test2_out_dir, 'visualizations')
    # os.makedirs(test2_vis_dir, exist_ok=True)
    # test2_metrics = os.path.join(test2_out_dir, 'metrics.pkl')
    # test2_cmd = [
    #     'python', 'tools/test.py',
    #     CONFIG["test2_config"],
    #     CONFIG["test2_checkpoint"],
    #     '--work-dir', test2_out_dir,
    #     '--show-dir', 'visualizations',
    #     '--out', test2_metrics
    # ]
    # # run_command(test2_cmd + ['--launcher', 'none'])

    # # 5. Comparison visualization
    # directories = {
    #     "Test1": test1_vis_dir,
    #     "Test2": test2_vis_dir,
    # }
    # run_comparison_from_config(
    #     directories=directories,
    #     output_dir=comparison_dir,
    #     show_image_width=CONFIG["show_image_width"],
    #     limit=CONFIG["comparison_limit"],
    #     save_comparisons=True
    # )

if __name__ == '__main__':
    main()