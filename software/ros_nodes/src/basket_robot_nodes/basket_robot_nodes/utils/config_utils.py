import logging

import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from .custom_exceptions import InvalidYamlConfig, SettingFileNotFound


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_settings(filepath: str) -> dict:
    """Load application settings from a YAML file."""
    try:
        with open(filepath, "r") as file:
            settings = yaml.safe_load(file)
            if not isinstance(settings, dict):
                raise InvalidYamlConfig(
                    "Settings file must contain a YAML dictionary at the top level."
                )
            logger.info(f"Settings loaded from {filepath}")
            return settings

    except FileNotFoundError as e:
        raise SettingFileNotFound(f"Settings file not found: {e}")

    except (ScannerError, ParserError) as e:
        raise InvalidYamlConfig(f"Error parsing settings file: {e}")
