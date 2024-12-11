from starlette.requests import Request
from ray.serve import metrics
from ray import serve
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
from utils.utils import TeacherModel

from transformers import AutoImageProcessor, AutoModelForImageClassification


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.8, "num_gpus": 0, "memory": 6 * 1024 * 1024 * 1024 }, name="student", route_prefix="/student")
class StudentClassifier:
    def __init__(self):
        # Pre-trained MobileNetV2 on ImageNet
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.eval()

        # Replace the classifier for CIFAR-10 (10 classes)
        # MobileNetV2's classifier output layer is at self.model.classifier[1]
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, 10)

        # CIFAR-10 label map
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Transforms (CIFAR-10 images are 32x32, but we resize to 224x224 to feed into MobileNet)
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
            "my_counter",
            description="Number of requests served by this deployment.",
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "mobilenet_v2_cifar10"})

    def predict(self, image_bytes: bytes) -> str:
        # Convert raw bytes to a PIL image and apply transforms
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Compute softmax probabilities and identify top prediction
        probabilities = F.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

        # Return only the predicted class name
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
        # Load the image processor and model from the teacher directory
        self.processor = AutoImageProcessor.from_pretrained("../teacher")
        self.model = AutoModelForImageClassification.from_pretrained("../teacher")
        self.model = TeacherModel()
        self.model.eval()

        # If the model config has id2label, use it. Otherwise, define your own labels.
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            # Convert the id2label dictionary into a list sorted by key
            self.class_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        else:
            # If no labels are defined in the config, fallback to CIFAR-10 classes or your own defined set.
            self.class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]

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

        # Process the image using the teacher’s processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
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
