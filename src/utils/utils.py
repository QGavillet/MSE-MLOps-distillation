import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset
from transformers import MobileNetV2Config, MobileNetV2ForImageClassification, ViTForImageClassification, \
    ViTImageProcessor
from torchvision.transforms import (
    Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, CenterCrop
)


def apply_train_transform(data):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    size = processor.size["height"]

    transform = transforms.Compose(
        [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
    )

    data['pixel_values'] = [transform(image.convert("RGB")) for image in data['img']]
    return data


def apply_test_transform(data):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    size = processor.size["height"]

    transform = transforms.Compose(
        [Resize(size), CenterCrop(size), ToTensor(), normalize]
    )

    data['pixel_values'] = [transform(image.convert("RGB")) for image in data['img']]
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

    train_ds.set_transform(apply_train_transform)
    test_ds.set_transform(apply_test_transform)

    return train_ds, test_ds


def collate_fn(data):
    pixel_values = torch.stack([sample["pixel_values"] for sample in data])
    labels = torch.tensor([sample["label"] for sample in data])
    return {"pixel_values": pixel_values, "labels": labels}


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
