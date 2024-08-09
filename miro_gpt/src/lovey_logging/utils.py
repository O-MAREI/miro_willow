from collections import defaultdict
import json
import logging
from typing import Dict, Union

def setup_logging(config_path: str, defaults: Dict[str, Union[str, int, float]] = {}) -> None:
    '''Sets up logging using the provided configuration json file'''
    with open(config_path) as f_in:
        config = json.load(f_in)

    defaults = defaultdict(str, **defaults)

    def format_config(config, defaults):
        '''Inserts default values if present and skips keys not in default'''
        for key, value in config.items():
            if isinstance(value, dict):
                format_config(value, defaults)
            elif isinstance(value, str):
                config[key] = value.format_map(defaults)

        return config

    config = format_config(config, defaults)
    logging.config.dictConfig(config)
