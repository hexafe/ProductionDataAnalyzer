from google.colab import files
import os
import shutil
from pathlib import Path
from typing import Union, List
from tempfile import TemporaryDirectory

class ColabAdapter:
    def get_available_memory(self) -> float:
        """
        Get available system memory in GB
        
        Returns:
            float: Available memory in GB
        """
        mem_info = os.popen('free -g').readlines()[1].split()
        return int(mem_info[6]) / 1e3  # Convert KB to GB

    def is_directory(self, path: Union[str, Path]) -> bool:
        """
        Check if path is a directory
        
        Args:
            path (Union[str, Path]): Path to check
            
        Returns:
            bool: True if directory exists
        """
        return Path(path).is_dir()
    def upload_files(self) -> List[Path]:
        """
        Handle file uploads in Colab environment
        
        Returns:
            List[Path]: List of uploaded file paths in temporary directory
            
        Raises:
            RuntimeError: If upload fails or no files are selected
        """
        with TemporaryDirectory() as tmp_dir:
            uploaded = files.upload()
            if not uploaded:
                raise RuntimeError("No files uploaded")
                
            file_paths = []
            for filename, content in uploaded.items():
                file_path = Path(tmp_dir) / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                file_paths.append(file_path)
            return file_paths

    def extract_archive(self, archive_path: Path) -> Path:
        """
        Extract archive files in Colab environment
        
        Args:
            archive_path (Path): Path to archive file
            
        Returns:
            Path: Directory containing extracted files
            
        Raises:
            ValueError: For unsupported archive formats
        """
        extract_dir = archive_path.parent / f"{archive_path.stem}_extracted"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            shutil.unpack_archive(str(archive_path), extract_dir)
        except shutil.ReadError:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
        return extract_dir

    def cleanup(self, path: Path) -> None:
        """
        Clean up temporary files in Colab
        
        Args:
            path (Path): Directory or file to remove
        """
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
