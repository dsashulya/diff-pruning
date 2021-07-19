from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import NoReturn

import numpy as np

LOG_DIR = Path.cwd() / 'runs'


def prepare_log_dir(model_name: str, base_dir=LOG_DIR):
    now = datetime.now()
    log_dir = base_dir / f'{model_name}-{now:%Y%m%d-%H%M-%S}'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(model_name: str) -> SummaryWriter:
    log_dir = prepare_log_dir(model_name=model_name)
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer


def write_params(writer, args):
    output = f'Model path {args.model_name_or_path}  \n'
    output += f'Task name {args.task_name}  \n'
    output += f'Epochs {args.num_train_epochs}  \n'
    output += f'W learning rate {args.lr_params}  \n'
    output += f'Alpha LR {args.lr_alpha}  \n'
    output += f'Sparsity penalty {args.sparsity_penalty}  \n'
    output += f'Alpha init {args.alpha_init}  \n'
    output += f'Weight decay {args.weight_decay}  \n'
    output += f'Batch size {args.batch_size}  \n'
    output += f'Per params alpha {args.per_params_alpha}  \n'
    output += f'Per layer alpha {args.per_layer_alpha}  \n'
    output += f'Concrete lower {args.concrete_lower}  \n'
    output += f'Concrete upper {args.concrete_upper}  \n'
    output += f'Start update steps {args.update_steps_start}'
    writer.add_text('Parameters', output, args.update_steps_start)


def log_start_training(logger, args, train_dataloader, t_total) -> NoReturn:
    logger.info(f"***** Running training *****")
    logger.info(f"  Model name = {args.model_name}")
    logger.info(f"  Diff mode = {not args.no_diff}")
    logger.info(f"  Num batches = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f"  Weight decay = {args.weight_decay}")
    logger.info(f"  Concrete lower = {args.concrete_lower}")
    logger.info(f"  Concrete upper = {args.concrete_upper}")
    if not args.no_diff:
        logger.info(f"  Params LR = {args.lr_params}, Alpha LR {args.lr_alpha}")
        logger.info(f"  Sparsity penalty = {args.sparsity_penalty}")
        logger.info(f"  Per params alpha = {args.per_params_alpha}")
        logger.info(f"  Per layer alpha = {args.per_layer_alpha}")
    else:
        logger.info(f"  Params LR = {args.lr_params}")


def log_epoch(logger, epoch: int, update_steps: int, train_losses: list,
              val_losses: list, nonzero_params: list = None,
              l0_penalties: list = None, no_diff=False) -> NoReturn:
    string = f"Epoch {epoch + 1} | Update step {update_steps} average values:" + \
             f"\n\ttrain loss = {np.array(train_losses).mean():.3f} "
    if len(val_losses):
        string += f"val loss = {np.array(val_losses).mean():.3f}"

    if not no_diff:
        string += f"\n\tnonzero params = {nonzero_params:,.0f}" + \
                  f"\n\tL0 penalty = {np.array(l0_penalties).mean():,.3f}"
    logger.info(string)
