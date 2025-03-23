from pathlib import Path
from typing import List
from .base_strategy import BaseLoader
from ..adapters import get_adapter

class ArchiveHandler(BaseLoader):
    """Archive loading strategy with recursive processing"""
    supported_extensions = ['.zip', '.7z', '.rar', '.tar', '.gz']
    
    def __init__(self, strategies: List[BaseLoader]):
        self.adapter = get_adapter()
        self.strategies = strategies

    def load(self, file_path: Path) -> List[Path]:
        extract_dir = self.adapter.extract_archive(file_path)
        return self._process_extracted_files(extract_dir)

    def _process_extracted_files(self, directory: Path) -> List[Path]:
        processed = []
        for strategy in self.strategies:
            for file in directory.rglob('*'):
                if strategy.supports(file):
                    processed.append(strategy.load(file))
        return processed
