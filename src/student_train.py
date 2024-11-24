import shutil
import ray.train as train
import ray.train.torch
import os
import tempfile
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
import ray.train.torch
from ray.train import ScalingConfig
from utils.utils import load_train_data, StudentModel, get_scaling_config
from utils.utils import TeacherModel
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
    teacher = config["teacher"]
    student = StudentModel()
    teacher = ray.train.torch.prepare_model(teacher)
    student = ray.train.torch.prepare_model(student)

    teacher.eval()

    # Transform and DataLoader
    train_data = load_train_data()
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Precompute teacher predictions
    teacher_predictions = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(ray.train.torch.get_device())
            teacher_outputs = teacher(images)
            teacher_predictions.extend(teacher_outputs.cpu())

    teacher_predictions = torch.stack(teacher_predictions).to(ray.train.torch.get_device())

    # Student optimizer
    student_optimizer = Adam(student.parameters(), lr=config["lr"])

    # Training loop
    epochs = config["epochs"]
    temperature = config["temperature"]
    alpha = config["alpha"]

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(ray.train.torch.get_device()), labels.to(ray.train.torch.get_device())
            teacher_preds_batch = teacher_predictions[i * config["batch_size"]:(i + 1) * config["batch_size"]]

            student_optimizer.zero_grad()
            outputs = student(images)
            loss = distillation_loss(outputs, labels, teacher_preds_batch, temperature, alpha)
            loss.backward()
            student_optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(student.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            ray.train.report(
                metrics={"loss": avg_loss, "epoch": epoch},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )


# Configure scaling and resource requirements
scaling_config = get_scaling_config()

# Initialize teacher model and pass it to the training function
teacher_model = TeacherModel()
teacher_model = torch.nn.DataParallel(teacher_model)
teacher_model.load_state_dict(torch.load('./models/teacher/model.pt'))  # Load pre-trained teacher model

# Launch distributed training with Ray
trainer = ray.train.torch.TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    train_loop_config={
        "teacher" : teacher_model,
        "epochs": 1,
        "lr": 0.001,
        "batch_size": 64,
        "temperature": 3,
        "alpha": 0.8,
    }
)


result = trainer.fit()

# Ensure the model folder exists
model_save_path = "./models/student"
os.makedirs(model_save_path, exist_ok=True)

# Copy the trained model to ./model
with result.checkpoint.as_directory() as checkpoint_dir:
    model_file_path = os.path.join(model_save_path, "model.pt")
    shutil.copyfile(os.path.join(checkpoint_dir, "model.pt"), model_file_path)

print(f"Training completed. Model saved to {model_file_path}")