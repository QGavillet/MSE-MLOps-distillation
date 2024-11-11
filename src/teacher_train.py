import pickle
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from ray import train, tune
from ray.train import Checkpoint
from utils.utils import Net, load_test_data, load_data


def train_cifar(config):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    if config["smoke_test"]:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

            # calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        avg_training_loss = running_loss / epoch_steps
        train_accuracy = correct_train / total_train

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total_val = 0
        correct_val = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Calculate validation accuracy
        val_accuracy = correct_val / total_val

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {
                    "train_loss": avg_training_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": (val_loss / val_steps),
                    "val_accuracy": val_accuracy
                },
                checkpoint=checkpoint,
            )
    print("Finished Training")


if __name__ == "__main__":
    config = {
        "l1": tune.grid_search([4]),  # tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.grid_search([4, 8]),  # tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.grid_search([1e-2]),  # tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.grid_search([2, 4]),
        "epochs": 10,
        "smoke_test": True,
    }

    tuner = tune.Tuner(
        train_cifar,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    model_path = "models/teacher"
    os.makedirs(model_path, exist_ok=True)

    # save best result pkl
    with open(model_path + "/model.pkl", "wb") as f:
        pickle.dump(best_result, f)

    # move file
    os.rename(best_result.path + "/progress.csv", model_path + "/progress.csv")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["val_accuracy"]))