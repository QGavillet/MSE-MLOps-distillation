import argparse
import shutil
from datetime import datetime
from typing import Dict

import pytz
import ray.train as train
import ray.train.torch
import os
import tempfile
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
import ray.train.torch
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from utils.utils import load_train_data, get_scaling_config, set_seed
from utils.utils import TeacherModel

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_func(config):
    model = TeacherModel()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    train_data = load_train_data()
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    wb = setup_wandb(project="MSE-MLOps-distillation", trial_name=config["exp_name"], rank_zero_only=False)

    wb_config = config.copy()
    wb_config.pop("exp_name")
    wb.log({"config": wb_config})

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            ray.train.report(
                metrics={"loss": avg_loss, "epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
            wb.log({"loss": avg_loss})


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a teacher model")
    parser.add_argument("--output_folder", type=str, default="./models", help="Folder to save the model")
    args = parser.parse_args()

    set_seed(16)

    ray.init()

    # Configure scaling and resource requirements
    scaling_config = get_scaling_config()

    now = datetime.now(tz=pytz.timezone('Europe/Zurich'))
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "teacher_train_" + now
    config = {
        "epochs": 1,
        "lr": 0.001,
        "batch_size": 64,
        "exp_name": exp_name,
    }

    # Launch distributed training
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config=config,
        run_config=train.RunConfig(
            failure_config=train.FailureConfig(1),
        ),
    )
    result = trainer.fit()

    # Ensure the model folder exists
    model_save_path = args.output_folder
    os.makedirs(model_save_path, exist_ok=True)

    # Copy the trained model to ./model
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_file_path = os.path.join(model_save_path, "teacher.pt")
        shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)


    print(f"Training completed. Model saved to {model_file_path}")
