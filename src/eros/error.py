class ErosError(Exception):
    """Base exception for all eros errors."""

class TaggerError(ErosError):
    """Raised when there is an error with the tagger."""

class VideoError(ErosError):
    """Raised when there is an error processing a video."""

class ImageError(ErosError):
    """Raised when there is an error processing an image."""

class DatabaseError(ErosError):
    """Raised when there is an error with the database."""
