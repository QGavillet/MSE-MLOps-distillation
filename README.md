# MLOps distillation project

## Setup

Create a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the following content:
```
export WANDB_API_KEY="your_wandb_api_key"
```

Create a `gcs_credentials.json` file with the credentials to access the google cloud storage.


# Run an experiment
1. Forward the ray server port to the local machine. For this you need to have the kubernetes config with access to the cluster.
```bash
kubectl port-forward service/raycluster-kuberay-head-svc 10001:10001
```
* To have access to the dashboard, you need to forward the port 8265 as well.
```bash
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

2. Run the experiment
```bash
dvc repro
```

3. Push to the gc storage.
```bash
dvc push
```
