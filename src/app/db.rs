use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;

use super::file::TaggingResultSimple;

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
                tags TEXT NOT NULL,
                rating TEXT
            )",
            [],
        )?;
        Ok(())
    }

    pub fn save_image_tags_batch(&mut self, results: &[TaggingResultSimple]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for result in results {
            tx.execute(
                "INSERT OR REPLACE INTO images (filename, size, hash, tags, rating) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![result.filename, result.size, result.hash, result.tags, result.rating],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}