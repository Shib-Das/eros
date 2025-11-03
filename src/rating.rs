//! # Rating Module
//!
//! This module provides the `RatingModel` for content rating of images.
//!
//! The `RatingModel` uses a pretrained ONNX model to classify images as "safe" or "nsfw".
//! It handles the downloading of the model and its configuration from the Hugging Face Hub,
//! image preprocessing, and inference.
//!
//! The main components are `RatingModel` for managing the rating process and `Rating`
//! for representing the classification result.

use anyhow::{Context, Result};
use image::DynamicImage;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{
    file::{RatingConfigFile, RatingModelFile, RatingPreprocessorConfigFile},
    processor::{ImagePreprocessor, ImageProcessor},
};

/// The result of a rating operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rating {
    Nsfw,
    Sfw,
}

impl Rating {
    /// Creates a new `Rating` from a label string.
    fn from_label(label: &str) -> Result<Self> {
        match label {
            "nsfw" => Ok(Rating::Nsfw),
            "sfw" => Ok(Rating::Sfw),
            _ => anyhow::bail!("Unknown rating label: {}", label),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Rating::Nsfw => "nsfw",
            Rating::Sfw => "sfw",
        }
    }
}

/// The configuration for the rating model.
#[derive(Debug, Deserialize)]
struct RatingModelConfig {
    id2label: HashMap<String, String>,
}

impl RatingModelConfig {
    /// Loads the configuration from a JSON file.
    async fn from_json(path: PathBuf) -> Result<Self> {
        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read rating model config at {:?}", path))?;
        let config: RatingModelConfig = serde_json::from_str(&content)
            .with_context(|| "Failed to deserialize rating model config")?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
struct RatingPreprocessorConfig {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    #[serde(rename = "size")]
    size: Size,
}

#[derive(Debug, Deserialize)]
struct Size {
    height: u32,
    width: u32,
}

impl RatingPreprocessorConfig {
    /// Loads the configuration from a JSON file.
    async fn from_json(path: PathBuf) -> Result<Self> {
        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read rating preprocessor config at {:?}", path))?;
        let config: RatingPreprocessorConfig = serde_json::from_str(&content)
            .with_context(|| "Failed to deserialize rating preprocessor config")?;
        Ok(config)
    }
}

/// A model for rating images as "safe" or "nsfw".
#[derive(Debug)]
pub struct RatingModel {
    session: Session,
    preprocessor: ImagePreprocessor,
    config: RatingModelConfig,
    input_name: String,
    output_name: String,
}

impl RatingModel {
    /// Creates a new `RatingModel`.
    pub async fn new() -> Result<Self> {
        let model_path = RatingModelFile::get().await?;
        let config_path = RatingConfigFile::get().await?;
        let preprocessor_config_path = RatingPreprocessorConfigFile::get().await?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)?;

        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();

        let preprocessor_config =
            RatingPreprocessorConfig::from_json(preprocessor_config_path).await?;
        let preprocessor = ImagePreprocessor::new(
            preprocessor_config.size.height,
            preprocessor_config.size.width,
            preprocessor_config.image_mean,
            preprocessor_config.image_std,
            false,
        );

        let config = RatingModelConfig::from_json(config_path).await?;

        Ok(Self {
            session,
            preprocessor,
            config,
            input_name,
            output_name,
        })
    }

    /// Rates a single image.
    pub fn rate(&mut self, image: &DynamicImage) -> Result<Rating> {
        let tensor = self.preprocessor.process(image)?;
        let value = Value::from_array(tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => value])?;

        let output_tensor = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;
        let probabilities = output_tensor.1;

        let argmax = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .context("Failed to find argmax of probabilities")?;

        let label = self
            .config
            .id2label
            .get(&argmax.to_string())
            .with_context(|| format!("Label not found for index: {}", argmax))?;

        Rating::from_label(label)
    }
}
