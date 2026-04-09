from .base import ModelRegistry

registry = ModelRegistry()


def register_model(name):
    def decorator(constructor):
        registry.register(name, constructor)
        return constructor
    return decorator
