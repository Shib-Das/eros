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

use anyhow::Result;
use image::DynamicImage;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{
    error::TaggerError,
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
    fn from_label(label: &str) -> Result<Self, TaggerError> {
        match label {
            "nsfw" => Ok(Rating::Nsfw),
            "sfw" => Ok(Rating::Sfw),
            _ => Err(TaggerError::Rating(format!("Unknown rating label: {}", label))),
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
    async fn from_json(path: PathBuf) -> Result<Self, TaggerError> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: RatingModelConfig = serde_json::from_str(&content)?;
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
    async fn from_json(path: PathBuf) -> Result<Self, TaggerError> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: RatingPreprocessorConfig = serde_json::from_str(&content)?;
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
    pub async fn new() -> Result<Self, TaggerError> {
        let model_path = RatingModelFile::get().await?;
        let config_path = RatingConfigFile::get().await?;
        let preprocessor_config_path = RatingPreprocessorConfigFile::get().await?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)?;

        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();

        let preprocessor_config = RatingPreprocessorConfig::from_json(preprocessor_config_path).await?;
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
    pub fn rate(&mut self, image: &DynamicImage) -> Result<Rating, TaggerError> {
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
            .unwrap();

        let label = self.config.id2label.get(&argmax.to_string()).ok_or_else(|| {
            TaggerError::Rating(format!("Label not found for index: {}", argmax))
        })?;

        Rating::from_label(label)
    }
}
