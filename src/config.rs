use crate::{
    error::TaggerError,
    file::{ConfigFile, PreprocessFile},
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_classes: u32,
    pub num_features: u32,
    pub pretrained_cfg: PretrainedCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedCfg {
    pub input_size: Vec<u32>, // [channels, height, width]
    pub fixed_input_size: bool,
    pub num_classes: u32,
}

impl ModelConfig {
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self, TaggerError> {
        let json = fs::read_to_string(config_path).map_err(|e| TaggerError::Io(e.to_string()))?;
        let config: ModelConfig =
            serde_json::from_str(&json).map_err(|e| TaggerError::Json(e.to_string()))?;
        Ok(config)
    }

    pub async fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let config_file = ConfigFile::new(repo_id).get().await?;
        Self::load(config_file)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    #[serde(rename = "type")]
    pub stage_type: String,
    pub size: Option<Vec<u32>>,
    pub mean: Option<Vec<f32>>,
    pub std: Option<Vec<f32>>,
}

impl PreprocessConfig {
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self, TaggerError> {
        let json = fs::read_to_string(config_path).map_err(|e| TaggerError::Io(e.to_string()))?;
        let config: PreprocessConfig =
            serde_json::from_str(&json).map_err(|e| TaggerError::Json(e.to_string()))?;
        Ok(config)
    }

    pub async fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
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
