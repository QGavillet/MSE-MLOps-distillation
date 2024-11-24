import shutil
import ray.train as train
import ray.train.torch
import os
import tempfile
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
import ray.train.torch

from utils.utils import load_train_data, get_scaling_config
from utils.utils import TeacherModel

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_func(config):
    # Initialize model, loss, and optimizer
    model = TeacherModel()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    train_data = load_train_data()
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

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



if __name__ == '__main__':
    ray.init()

    # Configure scaling and resource requirements
    scaling_config = get_scaling_config()

    # Launch distributed training
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config={
            "epochs" : 1,
            "lr" : 0.001,
            "batch_size" : 64
        }
    )
    result = trainer.fit()

    # Ensure the model folder exists
    model_save_path = "./models/teacher"
    os.makedirs(model_save_path, exist_ok=True)

    # Copy the trained model to ./model
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_file_path = os.path.join(model_save_path, "model.pt")
        shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)

    # result.metrics  # The metrics reported during training.
    # result.checkpoint  # The latest checkpoint reported during training.
    # result.path  # The path where logs are stored.
    # result.error  # The exception that was raised, if training failed.

    print(f"Training completed. Model saved to {model_file_path}")
