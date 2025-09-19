import os 
import yaml
from typing import Optional

SETTING_FILEPATH = "conf/setting.yaml"

def load_settings(filepath: str = SETTING_FILEPATH) -> dict:
    """Load application settings from a YAML file."""
    try:
        with open(filepath, 'r') as file:
            settings = yaml.safe_load(file)
            return settings
    except Exception as e:
        print(f"Failed to load settings from {filepath}: {e}")
        return 