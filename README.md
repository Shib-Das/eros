# Eros: An Image and Video Tagger

Eros is a command-line tool for tagging images and videos using ONNX models. It provides a terminal user interface (TUI) for an interactive experience, as well as a command-line interface (CLI) for scripting and automation.

## Features

- **TUI and CLI**: Use the interactive TUI or the scriptable CLI.
- **ONNX Runtime**: Powered by `ort` for efficient, cross-platform inference.
- **Image and Video Support**: Tag both images and videos.
- **Batch Processing**: Process multiple files at once.
- **Database Storage**: Stores tagging results in a SQLite database.

## Installation

You can install Eros using `cargo`:

```bash
cargo install --path .
```

## Usage

### TUI

To start the interactive TUI, run the following command:

```bash
eros
```

The TUI will guide you through the process of selecting directories, configuring the tagging pipeline, and processing your media files.

### CLI

The CLI provides a non-interactive way to use Eros. Here's an example of how to tag images in a directory:

```bash
eros --input-path ./images --threshold 0.6
```

#### Options

- `--model`: The name of the model to use (e.g., `SwinV2`).
- `--input-path`: The path to the directory containing the images to process.
- `--video-path`: The path to the directory containing the videos to process.
- `--threshold`: The confidence threshold for the tagger.
- `--batch-size`: The batch size for the tagger.
- `--show-ascii-art`: Show ASCII art previews of the images in the TUI.