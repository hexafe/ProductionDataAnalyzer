import sys
from .base_adapter import BaseAdapter
from .local_adapter import LocalAdapter

try:
    from .colab_adapter import ColabAdapter
except ImportError:
    ColabAdapter = None  # Define as None when not available

def detect_environment():
    """Detect current execution environment"""
    return 'colab' if 'google.colab' in sys.modules else 'local'

def get_adapter():
    """Get environment-appropriate adapter"""
    env = detect_environment()
    
    if env == 'colab':
        if ColabAdapter is None:
            raise RuntimeError("Colab dependencies not installed. Run: pip install google-colab gspread")
        return ColabAdapter()
    return LocalAdapter()

__all__ = ['BaseAdapter', 'LocalAdapter', 'ColabAdapter', 'get_adapter', 'detect_environment']
