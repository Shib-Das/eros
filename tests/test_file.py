import pytest
from pathlib import Path
from PIL import Image
from eros.file import discover_media, is_valid_image

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

def test_discover_media(test_directory):
    """Tests the discover_media function."""
    media_files = discover_media(test_directory)
    assert len(media_files) == 5
    assert {f.name for f in media_files} == {
        "image1.jpg",
        "image2.png",
        "image3.jpeg",
        "video1.mp4",
        "video2.avi",
    }

def test_discover_media_empty(tmp_path):
    """Tests discover_media with an empty directory."""
    d = tmp_path / "empty_dir"
    d.mkdir()
    media_files = discover_media(d)
    assert len(media_files) == 0

def test_discover_media_no_media(test_directory):
    """Tests discover_media with a directory containing no media files."""
    d = test_directory.parent / "no_media"
    d.mkdir()
    (d / "document.txt").touch()
    media_files = discover_media(d)
    assert len(media_files) == 0

def test_is_valid_image(tmp_path):
    """Tests the is_valid_image function."""
    # Create a valid image file
    valid_image_path = tmp_path / "valid.jpg"
    Image.new("RGB", (1, 1)).save(valid_image_path)
    assert is_valid_image(valid_image_path)

    # Create an invalid image file
    invalid_image_path = tmp_path / "invalid.jpg"
    invalid_image_path.write_text("not an image")
    assert not is_valid_image(invalid_image_path)

    # Test with a non-existent file
    assert not is_valid_image(tmp_path / "non_existent.jpg")
