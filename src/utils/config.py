import ray
from ray.train import ScalingConfig
from utils.utils import set_seed


def get_scaling_config():
    return ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 2})


def get_run_config():
    return ray.train.RunConfig(storage_path='gs://mlops-distillation', failure_config=ray.train.FailureConfig(0))


def get_ray_runtime_env():
    return {
        "working_dir": "./src",
        "conda": {
            "dependencies": [
                "pip",
                {
                    "pip": [
                        "torch==2.5.1",
                        "torchvision==0.20.1",
                        "wandb==0.18.7",
                        "transformers==4.47.0",
                        "ray[train]==2.38.0",
                        "matplotlib==3.9.2",
                        "datasets==3.1.0"
                    ]
                }
            ]
        }
    }


def setup():
    set_seed(16)


def get_wandb_api_key():
    return "d0e537cd5ce30632d57d1010332a2c397233b17b"
