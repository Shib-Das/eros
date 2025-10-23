import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch
from eros.db import Database, DatabaseError

@pytest.fixture
def db():
    """Creates a temporary in-memory database for testing."""
    db = Database(Path(":memory:"))
    db.connect()
    yield db
    db.close()

def test_connect(db):
    """Tests the connect method."""
    assert db.conn is not None
    assert isinstance(db.conn, sqlite3.Connection)

def test_create_tables(db):
    """Tests the _create_tables method."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
    assert cursor.fetchone() is not None

def test_add_and_get_tags(db):
    """Tests adding and retrieving tags."""
    file_path = Path("test.jpg")
    tags = {"tag1": 0.9, "tag2": 0.8}
    db.add_tags(file_path, tags)

    retrieved_tags = db.get_tags(file_path)
    assert retrieved_tags is not None
    assert retrieved_tags == tags

def test_get_tags_non_existent_file(db):
    """Tests retrieving tags for a file that doesn't exist."""
    retrieved_tags = db.get_tags(Path("non_existent.jpg"))
    assert retrieved_tags is None

@patch("sqlite3.connect")
def test_database_error(mock_sqlite_connect):
    """Tests that a DatabaseError is raised when the database connection fails."""
    # Configure the mock to raise an error
    mock_sqlite_connect.side_effect = sqlite3.Error("Test error")

    with pytest.raises(DatabaseError, match="Error connecting to database: Test error"):
        db = Database(Path("any/path/will/do"))
        db.connect()
