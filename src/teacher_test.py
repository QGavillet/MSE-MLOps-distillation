import argparse
import json
import os
from datetime import datetime
import pytz
from matplotlib import pyplot as plt
from utils.utils import load_test_data, load_train_data
from utils.utils import TeacherModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import wandb


# Evaluate the model
def evaluate_model(model):
    test_data = load_test_data()

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # generate confusion matrix
    class_names = test_data.classes
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names,  ha='right')
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
    plt.tight_layout()
    plt.title("Confusion Matrix")

    # log in wandb
    now = datetime.now(tz=pytz.timezone('Europe/Zurich'))
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "teacher_test_" + now
    wandb.init(project="MSE-MLOps-distillation", name=exp_name)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    wandb.log(metrics)
    wandb.log({"confusion_matrix": wandb.Image(plt)})

    # Save files
    folder_name = "metrics/teacher/"
    plt.savefig(folder_name + "confusion_matrix.png")
    with open(folder_name + "metrics.json", "w") as f:
        json.dump(metrics, f)


def create_dataset(model, data_path):
    train_data = load_train_data()

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    model.eval()
    outputs = []
    with torch.no_grad():
        for images, labels in train_loader:
            outputs.append(model(images))

    outputs = np.concatenate(outputs, axis=0)
    print(f"Dataset shape: {outputs.shape}")
    file_path = os.path.join(data_path, "teacher.npy")
    np.save(file_path, outputs)
    print(f"Dataset saved to {file_path}")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test a teacher model and create dataset for student model.")
    parser.add_argument("--model_path", type=str, default="./models/teacher.pt", help="Path to the teacher model.")
    parser.add_argument("--output_folder", type=str, default="./data", help="Folder to save the dataset")
    args = parser.parse_args()

    # Load the trained model
    trained_model = TeacherModel()
    trained_model = torch.nn.DataParallel(trained_model)
    trained_model.load_state_dict(torch.load(args.model_path))

    # Evaluate the model
    evaluate_model(trained_model)

    # Create dataset for the student model
    data_save_path = args.output_folder
    os.makedirs(data_save_path, exist_ok=True)
    create_dataset(trained_model, data_save_path)
