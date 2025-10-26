use anyhow::Result;
use eros::pipeline::TaggingResult;
use futures::stream::{self, StreamExt};
use serde::Serialize;
use std::path::{PathBuf};
use tokio::fs;

use crate::tag::fix_tag_underscore;

/// Supported image extensions.
pub const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];

/// Check if the path is an image file.
pub fn is_image(path: &str) -> Result<bool> {
    match PathBuf::from(path).extension() {
        Some(ext) => {
            let ext = ext.to_string_lossy().to_lowercase();
            Ok(IMAGE_EXTENSIONS.contains(&ext.as_str()))
        }
        None => Ok(false),
    }
}

/// Get image files from a directory.
pub async fn get_image_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir).await?;
    let mut tasks = vec![];

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let task = tokio::spawn(async move {
            if is_image(path.to_str().unwrap()).unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        });

        tasks.push(task);
    }

    let files = stream::iter(tasks)
        .buffer_unordered(16)
        .filter_map(|result| async move { result.ok().flatten() })
        .collect()
        .await;

    Ok(files)
}

#[derive(Serialize, Debug, Clone)]
pub struct TaggingResultSimpleTags {
    pub rating: String,
    pub character: Vec<String>,
    pub general: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct TaggingResultSimple {
    pub filename: String,
    pub size: u64,
    pub hash: String,
    pub tags: String,
    pub rating: String,
    pub tagger: TaggingResultSimpleTags,
}

impl From<TaggingResult> for TaggingResultSimpleTags {
    fn from(result: TaggingResult) -> Self {
        Self {
            rating: result
                .rating
                .first()
                .map_or("".to_string(), |(k, _)| k.clone()),
            character: result
                .character
                .keys()
                .map(|tag| fix_tag_underscore(tag))
                .collect(),
            general: result
                .general
                .keys()
                .map(|tag| fix_tag_underscore(tag))
                .collect(),
        }
    }
}

impl From<(TaggingResult, String, u64, String, String)> for TaggingResultSimple {
    fn from(
        (result, filename, size, hash, rating): (TaggingResult, String, u64, String, String),
    ) -> Self {
        let mut tags = result.character.keys().cloned().collect::<Vec<String>>();
        tags.extend(result.general.keys().cloned().collect::<Vec<String>>());

        let tags = tags
            .iter()
            .map(|tag| fix_tag_underscore(tag))
            .collect::<Vec<String>>()
            .join(", ");

        Self {
            filename,
            size,
            hash,
            tags,
            rating,
            tagger: TaggingResultSimpleTags::from(result),
        }
    }
}