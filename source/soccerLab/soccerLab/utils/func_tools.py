import inspect

def has_param(func, name: str) -> bool:
    sig = inspect.signature(func)
    return name in sig.parameters
