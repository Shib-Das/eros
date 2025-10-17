import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from .error import DatabaseError

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        """Connects to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self._create_tables()
            return self
        except sqlite3.Error as e:
            raise DatabaseError(f"Error connecting to database: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def _create_tables(self) -> None:
        """Creates the necessary tables for the application."""
        if not self.conn:
            raise DatabaseError("Database connection is not available.")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    score REAL NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
                """
            )
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error creating tables: {e}") from e

    def add_tags(self, file_path: Path, tags: Dict[str, float]) -> None:
        """Adds tags for a given file to the database."""
        if not self.conn:
            raise DatabaseError("Database connection is not available.")

        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO files (path) VALUES (?)", (str(file_path),))
            file_id = cursor.execute("SELECT id FROM files WHERE path = ?", (str(file_path),)).fetchone()[0]

            tag_data = [(file_id, tag, score) for tag, score in tags.items()]
            cursor.executemany("INSERT INTO tags (file_id, tag, score) VALUES (?, ?, ?)", tag_data)

            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error adding tags: {e}") from e

    def get_tags(self, file_path: Path) -> Optional[Dict[str, float]]:
        """Retrieves tags for a given file from the database."""
        if not self.conn:
            raise DatabaseError("Database connection is not available.")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT t.tag, t.score
                FROM tags t
                JOIN files f ON t.file_id = f.id
                WHERE f.path = ?
                """,
                (str(file_path),),
            )
            rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows} if rows else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Error getting tags: {e}") from e
