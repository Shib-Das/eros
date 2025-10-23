import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from eros.cli import tag

@pytest.fixture
def runner():
    """Creates a CliRunner for invoking CLI commands."""
    return CliRunner()

@pytest.fixture
def media_directory(tmp_path):
    """Creates a temporary directory with some dummy media files."""
    d = tmp_path / "media"
    d.mkdir()
    (d / "image1.jpg").touch()
    (d / "video1.mp4").touch()
    return d

@patch("eros.cli.ErosApp")
def test_tag_command_success(mock_eros_app_class, runner, media_directory):
    """Tests the tag command with valid options."""
    # Configure the mock ErosApp
    mock_app_instance = MagicMock()
    mock_eros_app_class.return_value = mock_app_instance

    # Invoke the tag command
    result = runner.invoke(
        tag,
        [
            "--input-path",
            str(media_directory),
            "--model-path",
            "tests/model.onnx",
            "--db-path",
            "test.db",
            "--threshold",
            "0.5",
            "--batch-size",
            "2",
        ],
    )

    # Assert that the command exited successfully
    assert result.exit_code == 0
    assert "Tagging complete." in result.output

    # Assert that the ErosApp was initialized and tag_media was called correctly
    mock_eros_app_class.assert_called_once_with(
        Path("tests/model.onnx"), Path("test.db"), 2
    )
    mock_app_instance.tag_media.assert_called_once_with(
        media_directory, 0.5
    )

def test_tag_command_missing_input_path(runner):
    """Tests the tag command with a missing input path."""
    result = runner.invoke(tag, ["--model-path", "tests/model.onnx"])
    assert result.exit_code != 0
    assert "Missing option '--input-path'" in result.output

def test_tag_command_missing_model_path(runner, media_directory):
    """Tests the tag command with a missing model path."""
    result = runner.invoke(tag, ["--input-path", str(media_directory)])
    assert result.exit_code != 0
    assert "Missing option '--model-path'" in result.output
