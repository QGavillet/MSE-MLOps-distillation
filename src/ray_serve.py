from starlette.requests import Request
from ray.serve import metrics
from ray import serve
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class MobileNetV2Classifier:
    def __init__(self):
        # Load a pre-trained MobileNetV2 model and set it to evaluation mode
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.eval()

        # Define image preprocessing transforms consistent with the model’s training
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            ),
        ])

        # Set up a counter metric for requests
        self.my_counter = metrics.Counter(
            "my_counter",
            description="Number of requests served by this deployment.",
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "mobilenet_v2"})

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

        # If you have a label map (e.g., a list of class names), you could map top_idx
        # to a human-readable label. For demonstration, we just return the index.
        # Example:
        # class_names = ["class0", "class1", ...] # Load or define these yourself
        # predicted_label = class_names[top_idx.item()]

        return f"Predicted class index: {top_idx.item()}, probability: {top_prob.item():.4f}"

    async def __call__(self, request: Request) -> str:
        # Expect the request body to contain the raw image bytes.
        image_bytes = await request.body()
        prediction = self.predict(image_bytes)
        self.my_counter.inc()
        return prediction


mobilenet_app = MobileNetV2Classifier.bind()
