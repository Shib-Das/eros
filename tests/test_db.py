from pathlib import Path
import pytest
from eros.db import Database

@pytest.fixture
def create_db(tmp_path: Path):
    """Creates a temporary in-memory database for testing."""
    db = Database(tmp_path / "test.db")
    db.connect()
    yield db
    db.close()

def test_add_and_get_tags(create_db: Database):
    """Tests that tags can be added and retrieved from the database."""
    file_path = Path("/path/to/image.jpg")
    tags = {"tag1": 0.9, "tag2": 0.8}
    create_db.add_tags(file_path, tags)

    retrieved_tags = create_db.get_tags(file_path)
    assert retrieved_tags == tags
