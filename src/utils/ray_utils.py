from ray.train import ScalingConfig


def get_scaling_config():
    return ScalingConfig(num_workers=2, use_gpu=False)


def get_ray_runtime_env():
    return {
        "pip": [
            "ray[data,train,tune,serve]==2.38.0",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "scipy==1.13.1",
            "scikit-learn==1.5.2",
            "wandb==0.18.7",
            "transformers==4.47.0"
        ]
    }
