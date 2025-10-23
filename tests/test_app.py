import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from eros.app import ErosApp

@pytest.fixture
def mock_db():
    """Mocks the Database class."""
    with patch("eros.app.Database") as mock:
        yield mock

@pytest.fixture
def mock_tagger():
    """Mocks the Tagger class."""
    with patch("eros.app.Tagger") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def media_directory(tmp_path):
    """Creates a temporary directory with some dummy media files."""
    d = tmp_path / "media"
    d.mkdir()
    (d / "image1.jpg").touch()
    (d / "video1.mp4").touch()
    return d

def test_eros_app_initialization(mock_tagger):
    """Tests that the ErosApp is initialized correctly."""
    app = ErosApp(Path("tests/model.onnx"), Path("test.db"), 2)
    mock_tagger.assert_called_once_with(str(Path("tests/model.onnx")), 2)
    assert app.db_path == Path("test.db")

def test_tag_media(mock_db, mock_tagger, media_directory):
    """Tests the tag_media method."""
    # Configure the mock Tagger
    mock_tagger_instance = mock_tagger.return_value
    mock_tagger_instance.batch_size = 2
    mock_tagger_instance.predict.return_value = [
        {"tag1": 0.9, "tag2": 0.8},
        {"tag3": 0.7, "tag4": 0.6},
    ]

    # Create an ErosApp instance
    app = ErosApp(Path("tests/model.onnx"), Path("test.db"), 2)

    # Call the tag_media method
    app.tag_media(media_directory, 0.5)

    # Assert that the predict method was called correctly
    assert mock_tagger_instance.predict.call_count == 1
    args, kwargs = mock_tagger_instance.predict.call_args
    assert len(args[0]) == 2
    assert kwargs == {"size": 224}

    # Assert that the add_tags method was called correctly
    mock_db_instance = mock_db.return_value.__enter__.return_value
    assert mock_db_instance.add_tags.call_count == 2
    mock_db_instance.add_tags.assert_any_call(
        media_directory / "image1.jpg", {"tag1": 0.9, "tag2": 0.8}
    )
    mock_db_instance.add_tags.assert_any_call(
        media_directory / "video1.mp4", {"tag3": 0.7, "tag4": 0.6}
    )

@patch("eros.app.Tagger")
def test_get_tags_for_media(mock_tagger_class, mock_db, media_directory):
    """Tests the get_tags_for_media method."""
    # Configure the mock Database
    mock_db_instance = mock_db.return_value.__enter__.return_value
    mock_db_instance.get_tags.return_value = ["tag1", "tag2"]

    # Create an ErosApp instance
    app = ErosApp(Path("tests/model.onnx"), Path("test.db"), 1)

    # Call the get_tags_for_media method
    tags = app.get_tags_for_media(media_directory / "image1.jpg")

    # Assert that the get_tags method was called correctly
    mock_db_instance.get_tags.assert_called_once_with(
        media_directory / "image1.jpg"
    )
    assert tags == ["tag1", "tag2"]
