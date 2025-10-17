# Eros: An Image and Video Tagger (Python Edition)

Eros is a command-line tool for tagging images and videos using ONNX models. It provides a terminal user interface (TUI) for an interactive experience, as well as a command-line interface (CLI) for scripting and automation. This is a complete rewrite of the original Rust project in Python.

## Features

- **TUI and CLI**: Use the interactive TUI or the scriptable CLI.
- **ONNX Runtime**: Powered by `onnxruntime` for efficient, cross-platform inference.
- **Image and Video Support**: Tag both images and videos.
- **Batch Processing**: Process multiple files at once.
- **Database Storage**: Stores tagging results in a SQLite database.

## Installation

You can install Eros using `poetry`:

```bash
poetry install
```

## Usage

### TUI

To start the interactive TUI, run the following command:

```bash
poetry run python -m src.eros.tui
```

The TUI will guide you through the process of selecting directories, configuring the tagging pipeline, and processing your media files.

### CLI

The CLI provides a non-interactive way to use Eros. Here's an example of how to tag images in a directory:

```bash
poetry run python -m src.eros.cli tag --input-path ./images --model-path model.onnx --threshold 0.6
```

#### Options

- `--model-path`: The path to the ONNX model.
- `--input-path`: The path to the directory containing the images to process.
- `--db-path`: The path to the SQLite database.
- `--threshold`: The confidence threshold for the tagger.
- `--batch-size`: The batch size for the tagger.
