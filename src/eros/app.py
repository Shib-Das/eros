from pathlib import Path
from typing import List
from .db import Database
from .file import discover_images
from .tagger import Tagger

class ErosApp:
    def __init__(self, model_path: Path, db_path: Path, batch_size: int):
        self.db = Database(db_path)
        self.tagger = Tagger(str(model_path), batch_size)

    def tag_images(self, input_path: Path, threshold: float):
        """Tags all images in the given path and stores the results in the database."""
        self.db.connect()
        try:
            image_paths = discover_images(input_path)
            for i in range(0, len(image_paths), self.tagger.batch_size):
                batch = image_paths[i : i + self.tagger.batch_size]
                results = self.tagger.predict(
                    [str(path) for path in batch], size=224
                )
                for path, result in zip(batch, results):
                    tags = {
                        tag: score
                        for tag, score in result.items()
                        if score >= threshold
                    }
                    if tags:
                        self.db.add_tags(path, tags)
        finally:
            self.db.close()

    def get_tags_for_image(self, image_path: Path):
        """Retrievels tags for a single image."""
        self.db.connect()
        try:
            return self.db.get_tags(image_path)
        finally:
            self.db.close()
