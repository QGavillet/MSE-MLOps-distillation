# MLOps distillation project

## Setup

Create a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run an experiment
For this you need to have access to the configured Google Cloud Storage bucket.

1. Forward the ray server port to the local machine. For this you need to have the kubernetes config with access to the cluster.
```bash
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 > port-forward.log 2>&1 &
```

2. Run the experiment
```bash
dvc repro
```

3. Push to the gc storage.
```bash
dvc push
```
