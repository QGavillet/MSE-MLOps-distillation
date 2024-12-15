import ray
from src.utils.config import get_scaling_config, get_run_config, setup, get_ray_runtime_env
from ray.air.integrations.wandb import setup_wandb
import ray.train.torch
import argparse
import shutil
from datetime import datetime
import yaml
import pytz
import os
import tempfile
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.utils.utils import load_data, TeacherModel, collate_fn

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")


def train_func(config):
    model = TeacherModel()
    model = model.to(device)
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    subset_size = config["subset_size"]
    if subset_size is None or subset_size == "None":
        subset_size = 6000
    train_data, val_data = load_data(subset_size=subset_size)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    wand_api_key = config["wandb_api_key"]
    config.pop("wandb_api_key")
    wb = setup_wandb(project="MSE-MLOps-distillation", trial_name=config["exp_name"], rank_zero_only=False,
                     config=config, api_key=wand_api_key)

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["pixel_values"]
            labels = batch["labels"]

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["pixel_values"]
                labels = batch["labels"]

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            ray.train.report(
                metrics={"loss": avg_train_loss, "val_loss": avg_val_loss},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
            wb.log({"loss": avg_train_loss, "val_loss": avg_val_loss})


def get_training_plot(df_history) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(df_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, df_history["loss"], label="Training loss")
    plt.plot(epochs, df_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a teacher model")
    parser.add_argument("--output_folder", type=str, default="./models", help="Folder to save the model")
    args = parser.parse_args()

    setup()

    # ray.init(address="ray://localhost:10001", runtime_env=get_ray_runtime_env())
    ray.init()

    now = datetime.now(tz=pytz.timezone('Europe/Zurich'))
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "teacher_train_" + now
    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train-teacher"]

    config = {
        "epochs": train_params["epochs"],
        "lr": train_params["lr"],
        "batch_size": train_params["batch_size"],
        "weight_decay": train_params["weight_decay"],
        "exp_name": exp_name,
        "subset_size": train_params["subset_size"],
        "wandb_api_key": os.environ.get("WANDB_API_KEY"),
    }

    # Launch distributed training
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=get_scaling_config(),
        train_loop_config=config,
        run_config=get_run_config(),
    )
    result = trainer.fit()

    # Ensure the model folder exists
    model_save_path = args.output_folder
    os.makedirs(model_save_path, exist_ok=True)

    # Copy the trained model to ./model
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_file_path = os.path.join(model_save_path, "teacher.pt")
        shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)

    # save metrics in json file
    metrics_folder = "metrics/teacher"
    os.makedirs(metrics_folder, exist_ok=True)

    fig = get_training_plot(result.metrics_dataframe)
    fig.savefig(os.path.join(metrics_folder, "training_plot.png"))

    print(f"Training completed. Model saved to {model_file_path}")
