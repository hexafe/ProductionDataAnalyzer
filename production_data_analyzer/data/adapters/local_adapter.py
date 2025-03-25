from pathlib import Path
from typing import List
from .base_adapter import BaseAdapter

class LocalAdapter(BaseAdapter):
    """Local filesystem adapter for CLI"""
    
    def get_available_memory(self):
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    
    def collect_files(self, path: Path) -> List[Path]:
        return [f for f in path.glob('*') if f.is_file()]
    
    def extract_archive(self, archive_path: Path) -> Path:
        import shutil
        extract_dir = archive_path.parent / f"{archive_path.stem}_extracted"
        shutil.unpack_archive(str(archive_path), extract_dir)
        return extract_dir
