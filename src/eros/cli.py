import click
from pathlib import Path
from .app import ErosApp

@click.group()
def cli():
    """Eros: An image and video tagger."""
    pass

@cli.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="The path to the directory containing the images to process.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The path to the ONNX model.",
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default="eros.db",
    help="The path to the SQLite database.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0, 1),
    default=0.6,
    help="The confidence threshold for the tagger.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="The batch size for the tagger.",
)
def tag(
    input_path: Path,
    model_path: Path,
    db_path: Path,
    threshold: float,
    batch_size: int,
):
    """Tags images in a directory."""
    app = ErosApp(model_path, db_path, batch_size)
    try:
        app.tag_images(input_path, threshold)
        click.echo("Tagging complete.")
    finally:
        app.close()

if __name__ == "__main__":
    cli()
