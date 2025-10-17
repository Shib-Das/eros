import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from eros.app import ErosApp

@pytest.fixture
def app():
    """Creates a temporary app with a mock tagger and an in-memory database."""
    with patch("eros.app.Tagger") as mock_tagger_class:
        # Configure the mock Tagger instance
        mock_tagger_instance = MagicMock()
        mock_tagger_instance.predict.return_value = [
            {"tag1": 0.9, "tag2": 0.8},
            {"tag3": 0.7, "tag4": 0.6},
        ]
        mock_tagger_instance.batch_size = 2
        mock_tagger_class.return_value = mock_tagger_instance

        # Initialize the app. The Tagger will be mocked.
        app = ErosApp(model_path=Path("dummy_model.onnx"), db_path=Path(":memory:"), batch_size=2)
        yield app

@pytest.fixture
def image_directory(tmp_path):
    """Creates a temporary directory with some dummy image files."""
    d = tmp_path / "images"
    d.mkdir()
    (d / "image1.jpg").touch()
    (d / "image2.png").touch()
    return d

def test_tag_images(app, image_directory):
    """Tests the tag_images method."""
    try:
        app.tag_images(image_directory, threshold=0.5)

        # Verify that the tagger's predict method was called
        app.tagger.predict.assert_called_once()

        # Verify that the tags were added to the database
        tags1 = app.get_tags_for_image(image_directory / "image1.jpg")
        tags2 = app.get_tags_for_image(image_directory / "image2.png")

        assert tags1 is not None
        assert "tag1" in tags1
        assert "tag2" in tags1

        assert tags2 is not None
        assert "tag3" in tags2
        assert "tag4" in tags2
    finally:
        app.close()

def test_get_tags_for_image(app, image_directory):
    """Tests the get_tags_for_image method."""
    try:
        # First, tag the images
        app.tag_images(image_directory, threshold=0.5)

        # Then, retrieve the tags for a single image
        tags = app.get_tags_for_image(image_directory / "image1.jpg")
        assert tags is not None
        assert tags == {"tag1": 0.9, "tag2": 0.8}
    finally:
        app.close()
