import argparse
from datetime import datetime

import pytz
from matplotlib import pyplot as plt
import numpy as np
from utils.utils import StudentModel
from utils.utils import load_test_data
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
    exp_name = "student_test_" + now
    wandb.init(project="MSE-MLOps-distillation", name=exp_name)
    wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    wandb.log({"confusion_matrix": wandb.Image(plt)})


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a student model with knowledge distillation.")
    parser.add_argument("--model_path", type=str, default="./models/student.pt", help="Path to the student model.")
    args = parser.parse_args()

    # Load the trained model
    trained_model = StudentModel()
    trained_model = torch.nn.DataParallel(trained_model)
    trained_model.load_state_dict(torch.load(args.model_path))

    # Evaluate the model
    evaluate_model(trained_model)
