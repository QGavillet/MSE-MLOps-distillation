import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import MobileNetV2Config, MobileNetV2ForImageClassification, ViTForImageClassification, \
    ViTImageProcessor
from torchvision.transforms import (
    Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, CenterCrop
)


def get_train_transform():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    size = processor.size["height"]

    return transforms.Compose(
        [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
    )


def get_test_transform():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    size = processor.size["height"]

    return transforms.Compose(
        [Resize(size), CenterCrop(size), ToTensor(), normalize]
    )


def load_train_data():
    transform = get_train_transform()

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    return train_data


def load_test_data():
    transform = get_test_transform()

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return test_data


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Define the Teacher Model
class TeacherModel(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        super(TeacherModel, self).__init__()
        id2label, label2id = get_label_maps()
        self.model = ViTForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id)

    def forward(self, x):
        return self.model(x).logits


# Define the student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Configure and initialize the MobileNetV2 model
        self.config = MobileNetV2Config(num_labels=10)
        self.model = MobileNetV2ForImageClassification(self.config)

    def forward(self, x):
        # Pass the input through the MobileNetV2 model
        return self.model(x).logits


def get_label_maps():
    id2label = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    label2id = {label: id for id, label in id2label.items()}
    return id2label, label2id
