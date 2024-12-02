# Steps to serve
### Build config

```bash
serve build src.ray_serve:translator_app -o ray_serve_config.yaml
```

where `src.ray_serve` is the path to the module containing the Ray Serve backend.

It then generates a ray_serve_config.yaml file and we need to add the working dir (a zip file of the code) and add the pip install in the runtime_env of our application. 

```yaml
  runtime_env:
    # Example with a release
    working_dir: "https://github.com/QGavillet/MSE-MLOps-distillation/archive/refs/tags/test.zip"
    pip:
      # Ref to github raw file
      - "-r https://raw.githubusercontent.com/QGavillet/MSE-MLOps-distillation/refs/heads/main/requirements.txt"
```

### Deploy
```bash
 serve deploy ray_serve_config.yaml
 ```

Deploy our newly create file to the ray cluster