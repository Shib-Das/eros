use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        Ok(Self { conn })
    }

    pub fn init(&self) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                tags TEXT NOT NULL
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                tags TEXT NOT NULL
            )",
            [],
        )?;
        Ok(())
    }

    pub fn save_image_tags(
        &self,
        filename: &str,
        size: u64,
        hash: &str,
        tags: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO images (filename, size, hash, tags) VALUES (?1, ?2, ?3, ?4)",
            params![filename, size, hash, tags],
        )?;
        Ok(())
    }

    pub fn save_video_tags(
        &self,
        filename: &str,
        size: u64,
        hash: &str,
        tags: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO videos (filename, size, hash, tags) VALUES (?1, ?2, ?3, ?4)",
            params![filename, size, hash, tags],
        )?;
        Ok(())
    }
}