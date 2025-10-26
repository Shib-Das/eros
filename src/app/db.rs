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
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS videos (
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

    pub fn save_image_tags(
    &mut self,
    filename: &str,
    size: u64,
    hash: &str,
    tags: &str,
    rating: &str,
) -> Result<()> {
    self.conn.execute(
        "INSERT OR REPLACE INTO images (filename, size, hash, tags, rating) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![filename, size, hash, tags, rating],
    )?;
    Ok(())
}

    pub fn save_video_tags_batch(&mut self, results: &[TaggingResultSimple]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for result in results {
            tx.execute(
                "INSERT OR REPLACE INTO videos (filename, size, hash, tags, rating) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![result.filename, result.size, result.hash, result.tags, result.rating],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn save_video_tags(
        &mut self,
        filename: &str,
        size: u64,
        hash: &str,
        tags: &str,
        rating: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO videos (filename, size, hash, tags, rating) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![filename, size, hash, tags, rating],
        )?;
        Ok(())
    }

    pub fn cleanup_video_tags(&mut self, hash: &str) -> Result<()> {
        let tags_string: String = self.conn.query_row(
            "SELECT tags FROM videos WHERE hash = ?1",
            params![hash],
            |row| row.get(0),
        )?;

        if tags_string.is_empty() {
            return Ok(());
        }

        let mut tags: Vec<&str> = tags_string.split(", ").filter(|s| !s.is_empty()).collect();
        tags.sort_unstable();
        tags.dedup();
        
        let new_tags_string = tags.join(", ");

        self.conn.execute(
            "UPDATE videos SET tags = ?1 WHERE hash = ?2",
            params![new_tags_string, hash],
        )?;

        Ok(())
    }
}