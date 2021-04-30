from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = Path.cwd() / 'train_logs'


def prepare_log_dir(model_name: str, base_dir=LOG_DIR):
    now = datetime.now()
    log_dir = base_dir / f'{model_name}-{now:%Y%m%d-%H%M-%S}'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(model_name: str) -> SummaryWriter:
    log_dir = prepare_log_dir(model_name=model_name)
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer
