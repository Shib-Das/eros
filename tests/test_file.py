from pathlib import Path
import pytest
from eros.file import discover_images, discover_videos

@pytest.fixture
def create_files(tmp_path: Path):
    """Creates a temporary directory with some image and video files."""
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.png").touch()
    (tmp_path / "vid1.mp4").touch()
    (tmp_path / "doc.txt").touch()
    return tmp_path

def test_discover_images(create_files: Path):
    """Tests that discover_images finds only image files."""
    images = discover_images(create_files)
    assert len(images) == 2
    assert {p.name for p in images} == {"img1.jpg", "img2.png"}

def test_discover_videos(create_files: Path):
    """Tests that discover_videos finds only video files."""
    videos = discover_videos(create_files)
    assert len(videos) == 1
    assert {p.name for p in videos} == {"vid1.mp4"}
