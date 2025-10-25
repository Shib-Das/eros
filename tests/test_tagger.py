import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

from eros.tagger import Tagger, TaggerError
from eros.error import ImageError

MODEL_PATH = Path("tests/model.onnx")

def test_tagger_initialization_valid():
    """Tests that the Tagger class initializes correctly with a valid model."""
    tagger = Tagger(str(MODEL_PATH))
    assert tagger.session is not None

def test_tagger_initialization_invalid():
    """Tests that a TaggerError is raised for an invalid model path."""
    with pytest.raises(TaggerError):
        Tagger("invalid/path/to/model.onnx")

@patch('onnxruntime.InferenceSession')
def test_predict(mock_inference_session):
    """
    Tests the predict method using a mock session to isolate the prediction logic.
    """
    # Configure the mock session
    mock_session_instance = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session_instance.get_inputs.return_value = [mock_input]
    mock_session_instance.get_outputs.return_value = [mock_output]

    # Dummy model output for a batch of 1
    dummy_output = np.random.rand(1, 10).astype(np.float32)
    mock_session_instance.run.return_value = [dummy_output]
    mock_inference_session.return_value = mock_session_instance

    # Initialize Tagger (model path doesn't matter as it's mocked)
    tagger = Tagger("dummy_model.onnx", batch_size=1)

    # Test prediction
    image_path = "tests/image.jpg"
    results = tagger.predict([image_path], size=224)

    # Assertions
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, dict)
    assert len(result) == 10  # 10 tags in the dummy output
    assert "tag_0" in result
    assert "tag_9" in result
    assert pytest.approx(result["tag_0"]) == dummy_output[0, 0]

    # Verify that the session's run method was called
    mock_session_instance.run.assert_called_once()

@patch('onnxruntime.InferenceSession')
def test_predict_batching(mock_inference_session):
    """
    Tests the predict method handles batching and padding correctly.
    """
    # --- Mock Setup ---
    mock_session_instance = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session_instance.get_inputs.return_value = [mock_input]
    mock_session_instance.get_outputs.return_value = [mock_output]

    # Dummy output for a batch size of 4
    dummy_output = np.random.rand(4, 10).astype(np.float32)
    mock_session_instance.run.return_value = [dummy_output]
    mock_inference_session.return_value = mock_session_instance

    # --- Test ---
    tagger = Tagger("dummy_model.onnx", batch_size=4)

    # Predict with 2 images (less than batch size, so padding is tested)
    image_paths = ["tests/image.jpg", "tests/image.jpg"]
    results = tagger.predict(image_paths, size=224)

    # --- Assertions ---
    assert len(results) == 2 # Should only return results for the actual images, not padding

    # Check the input that was passed to the ONNX session
    call_args, _ = mock_session_instance.run.call_args
    input_batch = call_args[1]['input']
    assert input_batch.shape[0] == 4 # The batch sent to the model should be padded to 4

@patch('onnxruntime.InferenceSession')
@patch('eros.tagger.hf_hub_download')
def test_tagger_download_model(mock_hf_hub_download, mock_inference_session):
    """
    Tests that the Tagger class attempts to download a model if it doesn't exist.
    """
    # Configure the mock hf_hub_download to return a dummy path
    mock_hf_hub_download.return_value = "downloaded/model.onnx"

    # Configure the mock InferenceSession to fail on the first call and succeed on the second
    mock_inference_session.side_effect = [
        NoSuchFile("File not found"),
        MagicMock()
    ]

    # Initialize Tagger with a path that should trigger a download
    tagger = Tagger("organization/repo_name/model.onnx")

    # Assertions
    mock_hf_hub_download.assert_called_once_with(repo_id="organization/repo_name", filename="model.onnx", local_dir="./models", local_dir_use_symlinks=False)
    assert mock_inference_session.call_count == 2
    assert tagger.session is not None
