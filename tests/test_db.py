import pytest
import sqlite3
from pathlib import Path
from eros.db import Database, DatabaseError

def test_connect():
    """Tests the connect method."""
    with Database(Path(":memory:")) as db:
        assert db.conn is not None
        assert isinstance(db.conn, sqlite3.Connection)

def test_create_tables():
    """Tests the _create_tables method."""
    with Database(Path(":memory:")) as db:
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
        assert cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
        assert cursor.fetchone() is not None

def test_add_and_get_tags():
    """Tests adding and retrieving tags."""
    file_path = Path("test.jpg")
    tags = {"tag1": 0.9, "tag2": 0.8}
    with Database(Path(":memory:")) as db:
        db.add_tags(file_path, tags)
        retrieved_tags = db.get_tags(file_path)
    assert retrieved_tags is not None
    assert retrieved_tags == tags

def test_get_tags_non_existent_file():
    """Tests retrieving tags for a file that doesn't exist."""
    with Database(Path(":memory:")) as db:
        retrieved_tags = db.get_tags(Path("non_existent.jpg"))
    assert retrieved_tags is None

def test_database_error():
    """Tests that a DatabaseError is raised for an invalid database path."""
    with pytest.raises(DatabaseError):
        with Database(Path("/invalid/path/to/db")):
            pass
