import logging
from typing import Dict, Any

class Context:
    """
    Context class to hold shared state and resources across the application.
    """
    def __init__(self, logger: logging.Logger, registry: Dict[str, Any]):
        self.logger = logger
        self.MODEL_REGISTRY = registry
        self.args = None

    # def register_model(self, name, data):
    #     self.registry[name] = data