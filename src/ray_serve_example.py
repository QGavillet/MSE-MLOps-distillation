from starlette.requests import Request
from ray.serve import metrics
from ray import serve
from transformers import pipeline


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")
        self.num_requests = 0
        self.my_counter = metrics.Counter(
            "my_counter",
            description=("The number of odd-numbered requests to this deployment."),
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "123"})

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator_app = Translator.bind()