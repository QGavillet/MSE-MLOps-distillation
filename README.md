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


## Run an experiment
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

## Steps to serve
### Build config

```bash
serve build src.ray_serve:translator_app -o ray_serve_config.yaml
```

where `src.ray_serve` is the path to the module containing the Ray Serve backend.

It then generates a ray_serve_config.yaml file and we need to add the working dir (a zip file of the code) and add the pip install in the runtime_env of our application. 

```yaml
  runtime_env:
    # Example with a release
    working_dir: "https://github.com/QGavillet/MSE-MLOps-distillation/releases/download/v27/code_and_models.zip"
    pip:
      # Ref to github raw file
      - "-r https://raw.githubusercontent.com/QGavillet/MSE-MLOps-distillation/refs/heads/main/requirements.txt"
```

### Port Forwarding
Essential step: we need to add the port forwarding to access our k8s cluster. For the deployment, we need to forward the `8265` port and for the testing it's the `8000` port.

### Deploy
Done automatically by the github action. But if you want to do it manually, you can run the following command:
```bash
 serve deploy ray_serve_config.yaml
 ```

Deploy our newly create file to the ray cluster

### Testing
To test our newly create example, run this python code:

```bash
python test/test_ray_serve.py
```
