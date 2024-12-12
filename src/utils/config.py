import ray
from dotenv import load_dotenv
from ray.runtime_env import RuntimeEnv
from ray.train import ScalingConfig
from utils.utils import set_seed


def get_scaling_config():
    return ScalingConfig(num_workers=1, use_gpu=False)


def get_run_config():
    return ray.train.RunConfig(storage_path='gs://mlops-distillation', failure_config=ray.train.FailureConfig(0))


def get_ray_runtime_env():
    return RuntimeEnv(
        pip={
            "packages": [
                "torch==2.5.1",
                "torchvision==0.20.1",
                "wandb==0.18.7",
                "transformers==4.47.0",
                "ray[train]==2.38.0",
                "matplotlib==3.9.2",
                "datasets==3.1.0"
            ],
        },
        py_modules=["utils.utils", "utils.config"],
    )


def setup():
    set_seed(16)
    load_dotenv()
