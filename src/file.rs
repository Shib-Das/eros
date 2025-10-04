use crate::error::TaggerError;
use anyhow::Result;
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const MODEL_ROOT: &str = "models";

pub async fn download_file(url: &str, dest_path: &Path) -> Result<(), TaggerError> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| TaggerError::Io(format!("Failed to create model directory: {}", e)))?;
    }

    let response = reqwest::get(url)
        .await
        .map_err(|e| TaggerError::Network(e.to_string()))?;

    if !response.status().is_success() {
        return Err(TaggerError::Network(format!(
            "Failed to download file: {} ({})",
            url,
            response.status()
        )));
    }

    let mut dest =
        File::create(&dest_path).map_err(|e| TaggerError::Io(format!("Failed to create file: {}", e)))?;

    let mut response = response;
    while let Some(chunk) = response.chunk().await.map_err(|e| TaggerError::Network(e.to_string()))? {
        dest.write_all(&chunk)
            .map_err(|e| TaggerError::Io(format!("Failed to write to file: {}", e)))?;
    }

    Ok(())
}

fn get_file_path(repo_id: &str, file_name: &str) -> PathBuf {
    PathBuf::from(MODEL_ROOT).join(repo_id).join(file_name)
}

pub async fn get(repo_id: &str, file_path: &str) -> Result<PathBuf, TaggerError> {
    let dest_path = get_file_path(repo_id, file_path);
    if dest_path.exists() {
        return Ok(dest_path);
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, file_path
    );

    download_file(&url, &dest_path).await?;

    Ok(dest_path)
}

/// Model for the Tagging
pub struct TaggerModelFile {
    repo_id: String,
    model_path: String,
}

impl TaggerModelFile {
    pub fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            model_path: "model.onnx".to_string(),
        }
    }

    pub async fn get(&self) -> Result<PathBuf, TaggerError> {
        get(&self.repo_id, &self.model_path).await
    }
}

/// CSV file that has the list of tags and ids.
pub struct TagCSVFile {
    repo_id: String,
    csv_path: String,
}

impl TagCSVFile {
    pub fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            csv_path: "selected_tags.csv".to_string(),
        }
    }

    pub async fn get(&self) -> Result<PathBuf, TaggerError> {
        get(&self.repo_id, &self.csv_path).await
    }
}

pub struct ConfigFile {
    repo_id: String,
    config_path: String,
}

impl ConfigFile {
    pub fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            config_path: "config.json".to_string(),
        }
    }

    pub async fn get(&self) -> Result<PathBuf, TaggerError> {
        get(&self.repo_id, &self.config_path).await
    }
}

pub struct PreprocessFile {
    repo_id: String,
    preprocess_path: String,
}

impl PreprocessFile {
    pub fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            preprocess_path: "preprocessor_config.json".to_string(),
        }
    }

    pub async fn get(&self) -> Result<PathBuf, TaggerError> {
        get(&self.repo_id, &self.preprocess_path).await
    }
}

const RATING_MODEL_REPO: &str = "AdamCodd/vit-base-nsfw-detector";

/// The ONNX model file for content rating.
pub struct RatingModelFile;
/// The model's configuration file.
pub struct RatingConfigFile;
/// The preprocessor's configuration file.
pub struct RatingPreprocessorConfigFile;

impl RatingModelFile {
    pub async fn get() -> Result<PathBuf, TaggerError> {
        get(RATING_MODEL_REPO, "onnx/model.onnx").await
    }
}

impl RatingConfigFile {
    pub async fn get() -> Result<PathBuf, TaggerError> {
        get(RATING_MODEL_REPO, "onnx/config.json").await
    }
}

impl RatingPreprocessorConfigFile {
    pub async fn get() -> Result<PathBuf, TaggerError> {
        get(RATING_MODEL_REPO, "onnx/preprocessor_config.json").await
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tokio::runtime::Runtime;

    fn run_async<F, T>(future: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        Runtime::new().unwrap().block_on(future)
    }

    #[test]
    fn test_get_model() {
        let repo_id = "SmilingWolf/wd-swinv2-tagger-v3";
        let model_file = TaggerModelFile::new(repo_id);
        let path = run_async(model_file.get()).unwrap();
        assert!(path.exists());
        assert_eq!(
            path,
            PathBuf::from("models/SmilingWolf/wd-swinv2-tagger-v3/model.onnx")
        );
    }

    #[test]
    fn test_get_tag_csv() {
        let repo_id = "SmilingWolf/wd-swinv2-tagger-v3";
        let tag_csv = TagCSVFile::new(repo_id);
        let path = run_async(tag_csv.get()).unwrap();
        assert!(path.exists());
        assert_eq!(
            path,
            PathBuf::from("models/SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv")
        );
    }

    #[test]
    fn test_get_config() {
        let repo_id = "SmilingWolf/wd-swinv2-tagger-v3";
        let config_file = ConfigFile::new(repo_id);
        let path = run_async(config_file.get()).unwrap();
        assert!(path.exists());
        assert_eq!(
            path,
            PathBuf::from("models/SmilingWolf/wd-swinv2-tagger-v3/config.json")
        );
    }

    #[test]
    fn test_get_rating_model() {
        let path = run_async(RatingModelFile::get()).unwrap();
        assert!(path.exists());
        assert_eq!(
            path,
            PathBuf::from("models/AdamCodd/vit-base-nsfw-detector/onnx/model.onnx")
        );
    }
}