from utils.utils import StudentModel
from utils.utils import load_test_data
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


# Load the trained model
trained_model = StudentModel()
trained_model = torch.nn.DataParallel(trained_model)
trained_model.load_state_dict(torch.load("./models/student/model.pt"))

# Evaluate the model
evaluate_model(trained_model)