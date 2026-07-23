import os

import yaml


def load_config(config_path="config/config.yaml"):
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
