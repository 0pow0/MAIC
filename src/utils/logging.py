from collections import defaultdict
import logging
import numpy as np
from tensorboardX.writer import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(logdir=directory_name)
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(self, args):
        if not WANDB_AVAILABLE:
            self.console_logger.warning("wandb is not installed. Install with: pip install wandb")
            return

        # Build config dict from args
        config = vars(args) if hasattr(args, '__dict__') else args

        # Get wandb settings from args
        project = getattr(args, 'wandb_project', 'MAIC')
        entity = getattr(args, 'wandb_entity', None)
        run_name = getattr(args, 'wandb_run_name', None)

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            reinit=True
        )
        self.use_wandb = True
        self.console_logger.info(f"Wandb initialized. Project: {project}, Run: {wandb.run.name}")

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.writer.add_scalar(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

        if self.use_wandb:
            wandb.log({key: value, "timestep": t}, step=t)

    def log_histogram(self, key, value, t):
        if self.use_tb:
            self.writer.add_histogram(key, value, t)

        if self.use_wandb:
            wandb.log({key: wandb.Histogram(value.cpu().numpy() if hasattr(value, 'cpu') else value)}, step=t)

    def log_embedding(self, key, value):
        if self.use_tb:
            self.writer.add_embedding(value, tag=key)

    def finish(self):
        if self.use_wandb:
            wandb.finish()

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            import torch as th
            item = "{:.4f}".format(th.mean(th.tensor([x[1] for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

