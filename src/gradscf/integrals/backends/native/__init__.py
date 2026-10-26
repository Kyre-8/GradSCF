"""Native CPU integral backend; mathematical derivative rules live in autodiff."""
from .ffi import evaluate

__all__ = ["evaluate"]
