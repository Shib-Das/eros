from pathlib import Path
from typing import List
from PIL import Image

def discover_media(path: Path) -> List[Path]:
    """
    Discovers all media files (images and videos) in the given path.

    Args:
        path: The path to the directory to search.

    Returns:
        A list of paths to the media files.
    """
    media_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
    ]
    return [
        f
        for f in path.glob("**/*")
        if f.is_file() and f.suffix.lower() in media_extensions
    ]

def is_valid_image(path: Path) -> bool:
    """
    Checks if a file is a valid image.

    Args:
        path: The path to the file.

    Returns:
        True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False
