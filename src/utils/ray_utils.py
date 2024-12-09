from ray.train import ScalingConfig


def get_scaling_config():
    return ScalingConfig(num_workers=2, use_gpu=False)
