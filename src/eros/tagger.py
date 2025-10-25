import onnxruntime as ort
import numpy as np
import os
from typing import List, Dict
from .error import TaggerError
from .processor import preprocess_image
from huggingface_hub import hf_hub_download

from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException, NoSuchFile

class Tagger:
    def __init__(self, model_path: str, batch_size: int = 1):
        try:
            self.session = ort.InferenceSession(model_path)
        except (RuntimeException, NoSuchFile) as e:
            if isinstance(e, NoSuchFile):
                try:
                    parts = model_path.split("/")
                    if len(parts) < 2:
                        raise ValueError("Invalid model path format. Expected 'repo_id/filename' or 'org/repo_id/filename'.")
                    filename = parts[-1]
                    repo_id = "/".join(parts[:-1])
                    local_dir = "./models"
                    os.makedirs(local_dir, exist_ok=True)
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
                    self.session = ort.InferenceSession(model_path)
                except Exception as download_e:
                    raise TaggerError(f"Error loading ONNX model: {e}, and failed to download it: {download_e}") from download_e
            else:
                raise TaggerError(f"Error loading ONNX model: {e}") from e
        self.batch_size = batch_size

    def predict(self, image_paths: List[str], size: int) -> List[Dict[str, float]]:
        """Predicts tags for a batch of images."""
        try:
            # Preprocess images in batches
            preprocessed_images = [preprocess_image(path, size) for path in image_paths]

            # Pad the batch if necessary
            if len(preprocessed_images) < self.batch_size:
                padding = [np.zeros_like(preprocessed_images[0])] * (self.batch_size - len(preprocessed_images))
                preprocessed_images.extend(padding)

            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            input_batch = np.stack(preprocessed_images)

            # Run inference
            outputs = self.session.run([output_name], {input_name: input_batch})

            # Process outputs
            results = []
            for i in range(len(image_paths)):
                result = {}
                for idx, score in enumerate(outputs[0][i]):
                    result[f"tag_{idx}"] = score
                results.append(result)

            return results

        except (ort.OrtError, TaggerError) as e:
            raise TaggerError(f"Error during inference: {e}") from e
