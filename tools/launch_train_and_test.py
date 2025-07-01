import os
import subprocess
from vizualise.comparator import run_comparison_from_config

"""
$env:PYTHONPATH = "D:\Sagi\GCP\GCP"
python tools/launch_train_and_test.py
"""

CONFIG = {
    "wandb_group": "multi_label_training",
    "wandb_name": "mask2former_e10_lre-5_kostanai_afs-multi-label_v1",
    "wandb_project": "building-segmentation-gcp",
    "train_config": "configs/gcp/mask2former_r50_kazgisa-kostanai-multi-label.py",
    "load_from": "checkpoints/mask2former_r50_pretrained_50e_whu-mix-vector.pth",
    # "resume_from": 'work_dir\mask2former_training\mask2former_e1_lre-5_kostanai_afs_quicktest\epoch_2.pth',
    "resume": False,  # Automatically resume from last available epoch if no resume_from/load_from specified
    "test1_config": "configs/gcp/gcp_r50_kazgisa-kostanai.py",
    "test1_checkpoint": "D:\Sagi\GCP\GCP\checkpoints\gcp_e5_lre-4_kostanai-afs_v1.pth",
    "test2_config": "configs/gcp/gcp_r50_kazgisa-kostanai.py",
    "test2_checkpoint": "checkpoints/gcp_r50_pretrained_12e_whu-mix-vector.pth",
    "gpus": 1,
    "comparison_limit": 30,
    "show_image_width": 10,
    "max_epochs": 10
}

def run_command(cmd):
    print(f"\n[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    run_dir = os.path.join('work_dir', CONFIG["wandb_group"], CONFIG["wandb_name"])
    os.makedirs(run_dir, exist_ok=True)
    comparison_dir = os.path.join(run_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Train
    train_cmd = [
        'python', 'tools/train.py',
        CONFIG["train_config"],
        '--wandb-group', CONFIG["wandb_group"],
        '--wandb-name', CONFIG["wandb_name"],
        '--wandb-project', CONFIG["wandb_project"]
    ]
    if CONFIG.get("resume_from"):
        train_cmd += ['--resume', CONFIG["resume_from"]]
    elif CONFIG.get("load_from"):
        train_cmd += ['--cfg-options', f'load_from={CONFIG["load_from"]}']
    elif CONFIG.get("resume"):
         train_cmd += ['--resume']
    run_command(train_cmd)

    # 2. Dynamically set test1_checkpoint to the last epoch checkpoint
    max_epochs = CONFIG.get("max_epochs", 10)
    checkpoint_name = f"epoch_{max_epochs}.pth"
    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    CONFIG["test1_checkpoint"] = checkpoint_path

    # 3. Test 1 (output dir: <wandb_name>_test)
    test1_out_dir = os.path.join(run_dir, 'test1')
    os.makedirs(test1_out_dir, exist_ok=True)
    test1_vis_dir = os.path.join(test1_out_dir, 'visualizations')
    os.makedirs(test1_vis_dir, exist_ok=True)
    test1_metrics = os.path.join(test1_out_dir, 'metrics.pkl')
    test1_cmd = [
        'python', 'tools/test.py',
        CONFIG["test1_config"],
        CONFIG["test1_checkpoint"],
        '--work-dir', test1_out_dir,
        '--show-dir', "visualizations",
        '--out', test1_metrics
    ]
    # run_command(test1_cmd + ['--launcher', 'none'])

    # 4. Test 2
    test2_out_dir = os.path.join(run_dir, 'test2')
    os.makedirs(test2_out_dir, exist_ok=True)
    test2_vis_dir = os.path.join(test2_out_dir, 'visualizations')
    os.makedirs(test2_vis_dir, exist_ok=True)
    test2_metrics = os.path.join(test2_out_dir, 'metrics.pkl')
    test2_cmd = [
        'python', 'tools/test.py',
        CONFIG["test2_config"],
        CONFIG["test2_checkpoint"],
        '--work-dir', test2_out_dir,
        '--show-dir', 'visualizations',
        '--out', test2_metrics
    ]
    # run_command(test2_cmd + ['--launcher', 'none'])

    # 5. Comparison visualization
    directories = {
        "Test1": test1_vis_dir,
        "Test2": test2_vis_dir,
    }
    # run_comparison_from_config(
    #     directories=directories,
    #     output_dir=comparison_dir,
    #     show_image_width=CONFIG["show_image_width"],
    #     limit=CONFIG["comparison_limit"],
    #     save_comparisons=True
    # )

if __name__ == '__main__':
    main()