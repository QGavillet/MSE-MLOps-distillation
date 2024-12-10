import ray
from ray.train import ScalingConfig
from utils.utils import set_seed


def get_scaling_config():
    return ScalingConfig(num_workers=2, use_gpu=False)


def get_run_config():
    return ray.train.RunConfig(failure_config=ray.train.FailureConfig(0))


def setup():
    set_seed(16)


def get_wandb_api_key():
    return "d0e537cd5ce30632d57d1010332a2c397233b17b"

