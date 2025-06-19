# MLflow Integration

This document explains how to use MLflow for experiment tracking in the GCP project.

## Installation

Install MLflow and its dependencies:

```bash
pip install -r requirements/mlflow.txt
```

## Usage

### 1. Training with MLflow Tracking

To enable MLflow tracking during training, use the `--mlflow` flag:

```bash
python tools/train.py configs/gcp/mask2former_r50_kazgisa-kostanai.py --mlflow
```

### 2. Customizing MLflow Settings

You can customize MLflow settings using command-line arguments:

```bash
python tools/train.py configs/gcp/mask2former_r50_kazgisa-kostanai.py \
    --mlflow \
    --mlflow-experiment "my-experiment" \
    --mlflow-tracking-uri "http://my-mlflow-server:5000"
```

### 3. Configuration in Config Files

You can also configure MLflow directly in your config files by adding the MLflow hook to `default_hooks`:

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=1,
        interval=1),
    mlflow=dict(
        type='MLflowLoggerHook',
        experiment_name='my-experiment',
        log_params=True,
        log_metrics=True,
        log_artifacts=True,
        log_interval=50
    ),
)
```

## What Gets Logged

### Parameters
- Model configuration (type, backbone, etc.)
- Training parameters (learning rate, batch size, epochs)
- Optimizer settings
- Dataset information

### Metrics
- Training loss and other metrics (logged every 50 iterations by default)
- Epoch-level metrics
- Validation metrics

### Artifacts
- Model checkpoints
- Configuration files
- Training logs

## Viewing Results

### 1. Local MLflow UI

Start the MLflow UI server to view your experiments:

```bash
python tools/start_mlflow_ui.py
```

Then open your browser and go to `http://localhost:5000`

### 2. Command Line

You can also view experiments using the MLflow CLI:

```bash
# List experiments
mlflow experiments list

# View runs in an experiment
mlflow runs list --experiment-id <experiment_id>

# Compare runs
mlflow runs compare --run-ids <run_id1> <run_id2>
```

## MLflow Hook Configuration

The `MLflowLoggerHook` supports the following parameters:

- `experiment_name` (str): Name of the MLflow experiment
- `log_params` (bool): Whether to log parameters (default: True)
- `log_metrics` (bool): Whether to log metrics (default: True)
- `log_artifacts` (bool): Whether to log artifacts (default: True)
- `log_interval` (int): Logging interval in iterations (default: 50)

## Remote MLflow Server

To use a remote MLflow server, set the tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

Or use the command-line argument:

```bash
python tools/train.py configs/gcp/mask2former_r50_kazgisa-kostanai.py \
    --mlflow \
    --mlflow-tracking-uri "http://your-mlflow-server:5000"
```

## Troubleshooting

### MLflow Not Installed
If you see a warning about MLflow not being installed, install it with:

```bash
pip install mlflow
```

### Connection Issues
If you're having trouble connecting to a remote MLflow server:
1. Check that the server is running
2. Verify the tracking URI is correct
3. Ensure network connectivity

### Permission Issues
If you encounter permission issues with artifact logging:
1. Check file permissions in your work directory
2. Ensure MLflow has write access to the artifact store

## Example Workflow

1. **Start training with MLflow:**
   ```bash
   python tools/train.py configs/gcp/mask2former_r50_kazgisa-kostanai.py --mlflow
   ```

2. **View results in MLflow UI:**
   ```bash
   python tools/start_mlflow_ui.py
   ```

3. **Compare different runs:**
   - Open the MLflow UI
   - Select multiple runs
   - Compare parameters, metrics, and artifacts

4. **Download artifacts:**
   ```bash
   mlflow artifacts download --run-id <run_id> --artifact-path checkpoints
   ``` 