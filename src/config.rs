//! # Configuration Management
//!
//! This module provides structs and functions for loading and managing model
//! and preprocessing configurations from Hugging Face repositories.

use crate::file::{ConfigFile, PreprocessFile};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Represents the main configuration for a model, typically loaded from `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The model's architecture (e.g., "SwinV2").
    pub architecture: String,
    /// The number of classes the model can predict.
    pub num_classes: u32,
    /// The number of features in the model's output.
    pub num_features: u32,
    /// The pretrained configuration for the model.
    pub pretrained_cfg: PretrainedCfg,
}

/// Represents the pretrained configuration of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedCfg {
    /// The input size of the model in `[channels, height, width]` format.
    pub input_size: Vec<u32>,
    /// Whether the input size is fixed.
    pub fixed_input_size: bool,
    /// The number of classes the model can predict.
    pub num_classes: u32,
}

impl ModelConfig {
    /// Loads a `ModelConfig` from a local file path.
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let json = fs::read_to_string(config_path)?;
        let config: ModelConfig = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Loads a `ModelConfig` from a Hugging Face repository.
    pub async fn from_pretrained(repo_id: &str) -> Result<Self> {
        let config_file = ConfigFile::new(repo_id).get().await?;
        Self::load(config_file)
    }
}

/// Represents the preprocessing configuration, typically loaded from `preprocessor_config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// The list of preprocessing stages.
    pub stages: Vec<Stage>,
}

/// Represents a single stage in the preprocessing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// The type of the stage (e.g., "resize", "normalize").
    #[serde(rename = "type")]
    pub stage_type: String,
    /// The size for resizing, if applicable.
    pub size: Option<Vec<u32>>,
    /// The mean values for normalization, if applicable.
    pub mean: Option<Vec<f32>>,
    /// The standard deviation values for normalization, if applicable.
    pub std: Option<Vec<f32>>,
}

impl PreprocessConfig {
    /// Loads a `PreprocessConfig` from a local file path.
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let json = fs::read_to_string(config_path)?;
        let config: PreprocessConfig = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Loads a `PreprocessConfig` from a Hugging Face repository.
    pub async fn from_pretrained(repo_id: &str) -> Result<Self> {
        let config_file = PreprocessFile::new(repo_id).get().await?;
        Self::load(config_file)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::file::ConfigFile;
    use std::fs;
    use tokio::runtime::Runtime;

    fn run_async<F, T>(future: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        Runtime::new().unwrap().block_on(future)
    }

    #[test]
    fn test_load_model_config_raw() {
        let config_file = run_async(ConfigFile::new("SmilingWolf/wd-swinv2-tagger-v3").get()).unwrap();
        let json = fs::read_to_string(config_file).unwrap();
        let _config: ModelConfig = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_load_model_config() {
        let config_file = run_async(ConfigFile::new("SmilingWolf/wd-swinv2-tagger-v3").get()).unwrap();
        let _config: ModelConfig = ModelConfig::load(&config_file).unwrap();
    }

    #[test]
    fn test_load_model_config_from_pretrained() {
        let _config =
            run_async(ModelConfig::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
    }

    #[test]
    fn test_load_model_config_from_pretrained_many() {
        let repo_ids = vec![
            "SmilingWolf/wd-eva02-large-tagger-v3".to_string(),
            "SmilingWolf/wd-vit-large-tagger-v3".to_string(),
            "SmilingWolf/wd-convnext-tagger-v3".to_string(),
        ];

        for repo_id in repo_ids {
            let _config = run_async(ModelConfig::from_pretrained(&repo_id));
            assert!(_config.is_ok(), "{}", repo_id);
        }
    }

}
