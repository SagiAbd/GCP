# Copyright (c) OpenMMLab. All rights reserved.
import wandb
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ValLossWandbHook(Hook):
    """Hook to log validation losses to wandb.
    
    This hook will log validation losses after each validation epoch.
    """
    
    def __init__(self, interval=1):
        self.interval = interval
        
    def after_val_epoch(self, runner: Runner) -> None:
        """Log validation losses after validation epoch."""
        if not self.every_n_epochs(runner, self.interval):
            return
            
        # Get validation losses from the log buffer
        log_vars = runner.log_buffer.get_log_vars()
        
        # Filter for loss-related metrics
        val_losses = {}
        for key, value in log_vars.items():
            if key.startswith('val_') and 'loss' in key.lower():
                val_losses[key] = value
            elif key.startswith('loss') and 'val' in key.lower():
                val_losses[key] = value
                
        # Also check for general loss metrics that might be validation losses
        for key, value in log_vars.items():
            if key.startswith('loss') and key not in val_losses:
                # This might be validation loss if we're in validation mode
                val_losses[f'val_{key}'] = value
                
        # Log to wandb
        if val_losses:
            wandb.log(val_losses, step=runner.epoch)
            
        # Also log the raw log_vars for debugging
        wandb.log({'val_log_vars': log_vars}, step=runner.epoch) 