from pathlib import Path
from typing import List

def discover_files(path: Path, extensions: List[str]) -> List[Path]:
    """Discovers all files with the given extensions in the given path."""
    return [
        f
        for f in path.glob("**/*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

def discover_images(path: Path) -> List[Path]:
    """Discovers all images in the given path."""
    return discover_files(path, [".jpg", ".jpeg", ".png", ".gif"])

def discover_videos(path: Path) -> List[Path]:
    """Discovers all videos in the given path."""
    return discover_files(path, [".mp4", ".mkv", ".avi", ".mov"])
