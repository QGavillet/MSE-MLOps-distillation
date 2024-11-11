import os
import pickle

import torch

from utils.utils import Net, load_test_data, load_data


def test_model(best_result, smoke_test=False):
    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    if smoke_test:
        _, testset = load_test_data()
    else:
        _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    # TODO take outputs to create dataset for the student model
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Best trial test set accuracy: {}".format(correct / total))


if __name__ == "__main__":
    with open("../models/teacher/model.pkl", "rb") as f:
        model = pickle.load(f)
    test_model(model, smoke_test=True)
