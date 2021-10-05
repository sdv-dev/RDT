"""Function to load addons."""

import glob
import importlib
import json
import os


def import_addons():
    """Import all the addon modules."""
    addons_path = os.path.dirname(os.path.realpath(__file__))
    imported = set()
    for addon_json_path in glob.glob(f'{addons_path}/*/*.json'):
        with open(addon_json_path, 'r', encoding='utf-8') as addon_json_file:
            transformers = json.load(addon_json_file).get('transformers', [])
            for transformer in transformers:
                transformer = transformer.split('.')
                module = '.'.join(transformer[:-1])
                if module not in imported:
                    importlib.import_module(module)
                    imported.add(module)
