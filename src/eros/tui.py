from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DirectoryTree, DataTable
from textual.containers import Container
from pathlib import Path
from .app import ErosApp

class ErosTUI(App):
    """An image and video tagger TUI."""

    def __init__(self, app: ErosApp):
        super().__init__()
        self.app = app

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container():
            yield DirectoryTree(".", id="dir-tree")
            yield DataTable(id="tag-table")
        yield Footer()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """Called when a file is selected in the directory tree."""
        table = self.query_one(DataTable)
        table.clear()
        tags = self.app.get_tags_for_image(Path(event.path))
        if tags:
            for tag, score in tags.items():
                table.add_row(tag, f"{score:.2f}")

if __name__ == "__main__":
    # This is just for running the TUI directly for development
    # In the final application, the app object will be created by the CLI
    from .db import Database
    from .tagger import Tagger

    # Create dummy objects for the TUI
    # In a real scenario, these would be initialized with the correct paths
    tagger = Tagger("model.onnx")
    db = Database(Path("eros.db"))
    app_instance = ErosApp(tagger, db)

    tui = ErosTUI(app_instance)
    tui.run()
