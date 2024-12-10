import shutil
from datetime import datetime
import argparse
import numpy as np
import pytz
import ray.train as train
import ray.train.torch
import os
import tempfile
import yaml
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ray.air.integrations.wandb import setup_wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
import ray.train.torch
from utils.utils import load_train_data, StudentModel, set_seed
from utils.ray_utils import get_scaling_config, get_run_config
import torch.nn.functional as F


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distillation loss function
def distillation_loss(y_student_pred, y_true, teacher_pred, temperature=3, alpha=0.5):
    hard_loss = nn.CrossEntropyLoss()(y_student_pred, y_true)
    soft_teacher_probs = F.softmax(teacher_pred / temperature, dim=1)
    soft_student_probs = F.log_softmax(y_student_pred / temperature, dim=1)
    soft_loss = F.kl_div(soft_student_probs, soft_teacher_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


# Ray training function
def train_func(config):
    # Initialize models and prepare for distributed training
    student = StudentModel()
    student = ray.train.torch.prepare_model(student)

    # Transform and DataLoader
    train_data = load_train_data()
    student_val_data = load_train_data()
    student_train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False)
    student_val_loader = DataLoader(student_val_data, batch_size=config["batch_size"], shuffle=False)
    student_train_loader = ray.train.torch.prepare_data_loader(student_train_loader)
    student_val_loader = ray.train.torch.prepare_data_loader(student_val_loader)

    # Load teacher dataset
    teacher_data_shard = train.get_dataset_shard("teacher_dataset")
    teacher_train_loader = teacher_data_shard.iter_torch_batches(
        batch_size=config["batch_size"], dtypes=torch.float32
    )

    # Student optimizer
    student_optimizer = Adam(student.parameters(), lr=config["lr"])

    # Training loop
    epochs = config["epochs"]
    temperature = config["temperature"]
    alpha = config["alpha"]

    # wb log
    wb = setup_wandb(project="MSE-MLOps-distillation", trial_name=config["exp_name"], rank_zero_only=False, config=config)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        # Iterate over both student and teacher loaders simultaneously
        for i, ((images, labels), teacher_preds_batch) in enumerate(zip(student_train_loader, teacher_train_loader)):
            images, labels = images.to(ray.train.torch.get_device()), labels.to(ray.train.torch.get_device())

            student_optimizer.zero_grad()
            student_outputs = student(images)
            teacher_output = teacher_preds_batch["data"]

            # Compute the distillation loss
            loss = distillation_loss(student_outputs, labels, teacher_output, temperature, alpha)
            loss.backward()
            student_optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(student_train_loader)

        # Validation loop
        student.eval()
        criterion = nn.CrossEntropyLoss()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in student_val_loader:
                outputs = student(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(student_val_loader)

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(student.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
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
    parser = argparse.ArgumentParser(description="Train a student model with knowledge distillation.")
    parser.add_argument("--dataset_path", type=str, default="./data/teacher.npy", help="Path to the teacher dataset.")
    parser.add_argument("--output_folder", type=str, default="./models", help="Folder to save the trained model.")
    args = parser.parse_args()

    set_seed(16)

    # Get absolute path of the teacher dataset
    teacher_data_path = os.path.abspath(args.dataset_path)
    teacher_data = np.load(teacher_data_path)
    teacher_dataset = ray.data.from_numpy(teacher_data)

    now = datetime.now(tz=pytz.timezone('Europe/Zurich'))
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "student_train_" + now
    train_params = yaml.safe_load(open("params.yaml"))["train-student"]

    # Launch distributed training with Ray
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=get_scaling_config(),
        train_loop_config={
            "epochs": train_params["epochs"],
            "lr": train_params["lr"],
            "batch_size": train_params["batch_size"],
            "temperature": train_params["temperature"],
            "alpha": train_params["alpha"],
            "exp_name": exp_name,
        },
        datasets={
            "teacher_dataset": teacher_dataset
        },
        run_config=get_run_config(),
    )

    result = trainer.fit()

    # Ensure the model folder exists
    model_save_path = args.output_folder
    os.makedirs(model_save_path, exist_ok=True)

    # Copy the trained model to ./model
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_file_path = os.path.join(model_save_path, "student.pt")
        shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)

    # save metrics in json file
    metrics_folder = "metrics/student"
    os.makedirs(metrics_folder, exist_ok=True)

    fig = get_training_plot(result.metrics_dataframe)
    fig.savefig(os.path.join(metrics_folder, "training_plot.png"))

    print(f"Training completed. Model saved to {model_file_path}")
