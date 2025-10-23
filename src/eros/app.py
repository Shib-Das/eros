from pathlib import Path
from typing import List
from .db import Database
from .file import discover_media
from .tagger import Tagger

class ErosApp:
    """The main application class for Eros."""

    def __init__(self, model_path: Path, db_path: Path, batch_size: int):
        """
        Initializes the ErosApp.

        Args:
            model_path: The path to the ONNX model.
            db_path: The path to the SQLite database.
            batch_size: The batch size for the tagger.
        """
        self.db_path = db_path
        self.tagger = Tagger(str(model_path), batch_size)

    def tag_media(self, input_path: Path, threshold: float) -> None:
        """
        Tags all media in the given path and stores the results in the database.

        Args:
            input_path: The path to the directory containing the media to process.
            threshold: The confidence threshold for the tagger.
        """
        media_paths = sorted(discover_media(input_path))
        with Database(self.db_path) as db:
            for i in range(0, len(media_paths), self.tagger.batch_size):
                batch = media_paths[i : i + self.tagger.batch_size]
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
                        db.add_tags(path, tags)

    def get_tags_for_media(self, media_path: Path) -> List[str]:
        """
        Retrieves tags for a single media file.

        Args:
            media_path: The path to the media file.

        Returns:
            A list of tags for the media file.
        """
        with Database(self.db_path) as db:
            return db.get_tags(media_path)
