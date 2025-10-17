import pytest
from pathlib import Path
from eros.file import discover_files, discover_images, discover_videos

@pytest.fixture
def test_directory(tmp_path):
    """Creates a temporary directory with a mix of files for testing."""
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "image1.jpg").touch()
    (d / "image2.png").touch()
    (d / "video1.mp4").touch()
    (d / "document.txt").touch()
    sub_dir = d / "sub"
    sub_dir.mkdir()
    (sub_dir / "image3.jpeg").touch()
    (sub_dir / "video2.avi").touch()
    return d

def test_discover_files(test_directory):
    """Tests the discover_files function."""
    files = discover_files(test_directory, [".jpg", ".png"])
    assert len(files) == 2
    assert {f.name for f in files} == {"image1.jpg", "image2.png"}

def test_discover_images(test_directory):
    """Tests the discover_images function."""
    images = discover_images(test_directory)
    assert len(images) == 3
    assert {i.name for i in images} == {"image1.jpg", "image2.png", "image3.jpeg"}

def test_discover_videos(test_directory):
    """Tests the discover_videos function."""
    videos = discover_videos(test_directory)
    assert len(videos) == 2
    assert {v.name for v in videos} == {"video1.mp4", "video2.avi"}

def test_discover_files_empty(tmp_path):
    """Tests discover_files with an empty directory."""
    d = tmp_path / "empty_dir"
    d.mkdir()
    files = discover_files(d, [".jpg"])
    assert len(files) == 0

def test_discover_images_no_images(test_directory):
    """Tests discover_images with a directory containing no images."""
    # Create a directory with only a text file
    d = test_directory.parent / "no_images"
    d.mkdir()
    (d / "document.txt").touch()
    images = discover_images(d)
    assert len(images) == 0
