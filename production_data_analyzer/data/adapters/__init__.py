import sys
from .base import BaseAdapter
from .colab_adapter import ColabAdapter
from .local_adapter import LocalAdapter

def detect_environment() -> str:
    """Detect execution environment"""
    return 'colab' if 'google.colab' in sys.modules else 'local'

def get_adapter() -> BaseAdapter:
    """Get environment-specific adapter"""
    return ColabAdapter() if detect_environment() == 'colab' else LocalAdapter()
