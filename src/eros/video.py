import ffmpeg
import numpy as np
from pathlib import Path
from typing import Iterator
from .error import VideoError

def extract_frames(video_path: Path, frame_rate: int = 1) -> Iterator[np.ndarray]:
    """Extracts frames from a video at the given frame rate."""
    try:
        # Get video information
        probe = ffmpeg.probe(str(video_path))
        video_info = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        if video_info is None:
            raise VideoError(f"No video stream found in {video_path}")

        width = video_info["width"]
        height = video_info["height"]

        # Set up the ffmpeg input stream
        in_stream = ffmpeg.input(str(video_path))

        # Set up the ffmpeg output stream
        out_stream = ffmpeg.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            r=str(frame_rate),
        )

        # Run the ffmpeg command
        process = ffmpeg.run_async(
            [in_stream, out_stream],
            pipe_stdout=True,
            pipe_stderr=True,
            quiet=True,
        )

        # Read frames from stdout
        while True:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield frame

        process.wait()

    except FileNotFoundError:
        raise VideoError(
            "ffmpeg not found. Please install ffmpeg and add it to your PATH."
        ) from None
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf8") if e.stderr else "Unknown error"
        raise VideoError(f"Error extracting frames: {stderr}") from e
