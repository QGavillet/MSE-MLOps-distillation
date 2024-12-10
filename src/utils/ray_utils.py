import ray
from ray.train import ScalingConfig


def get_scaling_config():
    return ScalingConfig(num_workers=2, use_gpu=False)

def get_run_config():
    return ray.train.RunConfig(failure_config=ray.train.FailureConfig(5))
