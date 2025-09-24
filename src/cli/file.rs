use anyhow::Result;
use eros::pipeline::TaggingResult;
use futures::stream::{self, StreamExt};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::tag::fix_tag_underscore;

/// Supported image extensions.
pub const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];
pub const VIDEO_EXTENSIONS: [&str; 4] = ["mp4", "mkv", "webm", "avi"];

/// Check if the path is a file or directory.
pub async fn is_file<P: AsRef<Path>>(path: P) -> Result<bool> {
    let metadata = fs::metadata(&path).await?;
    Ok(metadata.is_file())
}

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

/// Check if the path is a video file.
pub fn is_video(path: &str) -> Result<bool> {
    match PathBuf::from(path).extension() {
        Some(ext) => {
            let ext = ext.to_string_lossy().to_lowercase();
            Ok(VIDEO_EXTENSIONS.contains(&ext.as_str()))
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
            if is_image(path.to_str().unwrap()).unwrap() {
                Some(path)
            } else {
                None
            }
        });

        tasks.push(task);
    }

    let files = stream::iter(tasks)
        .buffer_unordered(16)
        .filter_map(|result| async move {
            match result {
                Ok(Some(path)) => Some(path),
                _ => None,
            }
        })
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
    pub tags: String,
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

impl From<TaggingResult> for TaggingResultSimple {
    fn from(result: TaggingResult) -> Self {
        let mut tags = result.character.keys().cloned().collect::<Vec<String>>();
        tags.extend(result.general.keys().cloned().collect::<Vec<String>>());

        let tags = tags
            .iter()
            .map(|tag| fix_tag_underscore(tag))
            .collect::<Vec<String>>()
            .join(", ");

        Self {
            tags,
            tagger: TaggingResultSimpleTags::from(result),
        }
    }
}