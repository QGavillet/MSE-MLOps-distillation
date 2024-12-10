# MLOps distillation project

## Setup
Create a runtime-env.yaml file in the root directory of the project and add the following environment variables:
```
conda:
  dependencies:
    - pip
    - pip:
        - torch==2.5.1
        - torchvision==0.20.1
        - wandb==0.18.7
        - transformers==4.47.0
        - "ray[train]==2.38.0"
        - matplotlib==3.9.2
        - datasets==3.1.0
        - python-dotenv
env:
  WAND_API_KEY: your_wandb_api_key
```

Create a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run an experiment
1. Forward the ray server port to the local machine. For this you need to have the kubernetes config with access to the cluster.
```bash
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 > port-forward.log 2>&1 &
```

2. Run the experiment
```bash
dvc repro
```

3. Push to the gc storage. For this you need to have the gcloud credentials set up.
```bash
dvc push
```
