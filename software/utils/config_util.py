import yaml
import logging  
from typing import Optional
from yaml.scanner import ScannerError
from yaml.parser import ParserError

SETTING_FILEPATH = "conf/setting.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(filepath: str = SETTING_FILEPATH) -> dict:
    """Load application settings from a YAML file."""
    try:
        with open(filepath, 'r') as file:
            settings = yaml.safe_load(file)
            logger.info(f"Settings loaded from {filepath}") 
            return settings
    except FileNotFoundError as e:
        logger.error(f"Settings file not found: {e}")
        return 
    except (ScannerError, ParserError) as e:
        logger.error(f"Error parsing settings file: {e}")
        return
    