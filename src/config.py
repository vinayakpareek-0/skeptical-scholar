import yaml
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
config_path = PROJECT_ROOT / "config.yaml"

def load_config(path:str=config_path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

