import shutil
import numpy as np
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
from utils.utils import load_train_data, StudentModel, get_scaling_config
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
    student_train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False)
    student_train_loader = ray.train.torch.prepare_data_loader(student_train_loader)

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

        avg_loss = running_loss / len(student_train_loader)

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(student.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            ray.train.report(
                metrics={"loss": avg_loss, "epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )


if __name__ == '__main__':
    ray.init()

    # Configure scaling and resource requirements
    scaling_config = get_scaling_config()

    # Get absolute path of the teacher dataset
    teacher_data_path = os.path.abspath("./data/teacher.npy")
    teacher_data = np.load(teacher_data_path)
    teacher_dataset = ray.data.from_numpy(teacher_data)

    # Launch distributed training with Ray
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config={
            "epochs": 1,
            "lr": 0.001,
            "batch_size": 64,
            "temperature": 3,
            "alpha": 0.8,
        },
        datasets={
            "teacher_dataset": teacher_dataset
        }
    )

    result = trainer.fit()

    # Ensure the model folder exists
    model_save_path = "./models"
    os.makedirs(model_save_path, exist_ok=True)

    # Copy the trained model to ./model
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_file_path = os.path.join(model_save_path, "student.pt")
        shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)

    print(f"Training completed. Model saved to {model_file_path}")
