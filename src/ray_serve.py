from starlette.requests import Request
from ray.serve import metrics
from ray import serve
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

import sys
import os
# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.utils.utils import TeacherModel
from src.student_train import SmallCNN


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.8, "num_gpus": 0, "memory": 6 * 1024 * 1024 * 1024}, name="student", route_prefix="/student")
class StudentClassifier:
    def __init__(self):
        # Load the student model using the provided checkpoint

        self.model = SmallCNN()
        checkpoint = torch.load("./models/student.pt", map_location="cpu")
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

        # Define CIFAR-10 classes
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Apply the same transforms as the teacher or other appropriate transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            ),
        ])

        self.my_counter = metrics.Counter(
            "student_model_counter",
            description="Number of requests served by this student deployment.",
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "student_model"})

    def predict(self, image_bytes: bytes) -> str:
        # Convert the image bytes into a PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        # Run inference with the student model
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            top_prob, top_idx = probabilities[0].max(dim=0)

        # Return the predicted class name
        predicted_label = self.class_names[top_idx.item()]
        return predicted_label

    async def __call__(self, request: Request) -> str:
        # Expect the request body to contain the raw image bytes.
        image_bytes = await request.body()
        prediction = self.predict(image_bytes)
        self.my_counter.inc()
        return prediction


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.8, "num_gpus": 0, "memory": 6 * 1024 * 1024 * 1024}, name="teacher", route_prefix="/teacher")
class TeacherClassifier:
    def __init__(self):
        # Load the teacher model using the provided checkpoint
        self.model = TeacherModel()
        checkpoint = torch.load("./models/teacher.pt", map_location="cpu")
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

        # Define CIFAR-10 classes
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Apply the same transforms as the student or appropriate transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            ),
        ])

        # Set up a counter metric for requests
        self.my_counter = metrics.Counter(
            "teacher_model_counter",
            description="Number of requests served by this teacher deployment.",
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "teacher_model"})

    def predict(self, image_bytes: bytes) -> str:
        # Convert the image bytes into a PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        # Run inference with the teacher model
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Assuming the TeacherModel outputs raw logits directly
            probabilities = F.softmax(outputs, dim=-1)
            top_prob, top_idx = probabilities[0].max(dim=0)

        # Return the predicted class name
        predicted_label = self.class_names[top_idx.item()]
        return predicted_label

    async def __call__(self, request: Request) -> str:
        # Expect the request body to contain the raw image bytes.
        image_bytes = await request.body()
        prediction = self.predict(image_bytes)
        self.my_counter.inc()
        return prediction


teacher_app = TeacherClassifier.bind()
student_app = StudentClassifier.bind()
