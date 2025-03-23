from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

class DataAdapter(ABC):
    @abstractmethod
    def collect_files(self) -> List[Path]:
        """Collect files from environment-specific sources"""
        pass
