import os
import yaml
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Default market settings if missing
        market_defaults = {
            "timezone": "America/Chicago",
            "weekly_open": "Sunday 16:00",
            "weekly_close": "Friday 16:00",
        }
        market = self._config.setdefault("market", {})
        for k, v in market_defaults.items():
            market.setdefault(k, v)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'signal.sequence_length')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_data_path(self, key: str) -> str:
        """Get a data path from the config, ensuring the directory exists"""
        path = self.get(f'data.{key}')
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

# Create a singleton instance
config = Config()
