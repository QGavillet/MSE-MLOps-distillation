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
from datasets import load_dataset
from matplotlib import pyplot as plt
from ray.air.integrations.wandb import setup_wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
import ray.train.torch
from transformers import ViTFeatureExtractor, ViTImageProcessor
from utils.config import get_scaling_config, get_run_config, setup, get_ray_runtime_env
import torch.nn.functional as F

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")


# Distillation loss function
def distillation_loss(y_student_pred, y_true, teacher_pred, temperature=3, alpha=0.5):
    hard_loss_fn = nn.CrossEntropyLoss()
    hard_loss = hard_loss_fn(y_student_pred, y_true)

    teacher_probs = F.softmax(teacher_pred / temperature, dim=1)
    soft_loss = F.kl_div(
        F.log_softmax(y_student_pred / temperature, dim=1),
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Dynamically compute the size of the flattened feature map
        self._flattened_size = self._get_flattened_size(input_size)

        self.fc1 = nn.Linear(self._flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, pixel_values,):
        x = self.pool(F.relu(self.conv1(pixel_values)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def apply_student_transform(data):
    feature_extractor = ViTImageProcessor.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
    inputs = feature_extractor(data["img"], return_tensors="pt")
    data["pixel_values"] = inputs["pixel_values"]
    return data


def load_data(subset_size=None):
    if subset_size is None or subset_size == "None":
        data = load_dataset('cifar10')
        train_ds = data['train']
        test_ds = data['test']
    else:
        train_size = int(subset_size * 0.8)
        test_size = subset_size - train_size
        train_split = 'train[:{}]'.format(train_size)
        test_split = 'test[:{}]'.format(test_size)
        train_ds, test_ds = load_dataset('cifar10', split=[train_split, test_split])

    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    train_ds.set_transform(apply_student_transform)
    test_ds.set_transform(apply_student_transform)

    return train_ds, test_ds


def collate_fn(data):
    pixel_values = torch.stack([sample["pixel_values"] for sample in data])
    labels = torch.tensor([sample["label"] for sample in data])
    return {"pixel_values": pixel_values, "labels": labels}


# Ray training function
def train_func(config):
    # Initialize models and prepare for distributed training
    student = SmallCNN()
    student = student.to(device)
    student = ray.train.torch.prepare_model(student)

    # Transform and DataLoader
    student_train_data, student_val_data = load_data(subset_size=config["subset_size"])
    student_train_loader = DataLoader(student_train_data, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    student_val_loader = DataLoader(student_val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
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
    wand_api_key = config["wandb_api_key"]
    config.pop("wandb_api_key")
    wb = setup_wandb(project="MSE-MLOps-distillation", trial_name=config["exp_name"], rank_zero_only=False,
                     config=config, api_key=wand_api_key)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        # Iterate over both student and teacher loaders simultaneously
        for i, (student_batch, teacher_preds_batch) in enumerate(zip(student_train_loader, teacher_train_loader)):
            images = student_batch["pixel_values"]
            labels = student_batch["labels"]

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
            for batch in student_val_loader:
                images = batch["pixel_values"]
                labels = batch["labels"]

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

    setup()

    #ray.init(address="ray://localhost:10001", runtime_env=get_ray_runtime_env())
    ray.init()

    # Get absolute path of the teacher dataset
    teacher_data_path = os.path.abspath(args.dataset_path)
    teacher_data = np.load(teacher_data_path)
    teacher_dataset = ray.data.from_numpy(teacher_data)

    now = datetime.now(tz=pytz.timezone('Europe/Zurich'))
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "student_train_" + now
    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train-student"]

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
            "subset_size": train_params["subset_size"],
            "wandb_api_key": os.environ.get("WANDB_API_KEY"),
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
