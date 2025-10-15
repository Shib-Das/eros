//! This module provides tools for preprocessing images before they are fed into a model.
//!
//! It defines the `ImageProcessor` trait for generic image processing operations
//! and provides a concrete implementation, `ImagePreprocessor`, which handles
//! resizing, padding, normalization, and color channel ordering.

use anyhow::{Context, Result};
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::{Array, Array3, Axis, Ix4};
use rayon::prelude::*;

use crate::config::{ModelConfig, PreprocessConfig};

/// A trait for processing images into tensors suitable for model input.
pub trait ImageProcessor {
    /// Processes a single image into a 4D tensor.
    fn process(&self, image: &DynamicImage) -> Result<Array<f32, Ix4>>;

    /// Processes a batch of images into a single 4D tensor.
    fn process_batch(&self, images: Vec<&DynamicImage>) -> Result<Array<f32, Ix4>>
    where
        Self: Sync,
    {
        let tensors: Result<Vec<_>> =
            images.into_par_iter().map(|img| self.process(img)).collect();
        let tensors = tensors?;

        ndarray::concatenate(
            Axis(0),
            &tensors.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )
        .context("Failed to concatenate tensors")
    }
}

/// A preprocessor that resizes, pads, and normalizes images.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    pub height: u32,
    pub width: u32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub bgr: bool,
}

impl ImagePreprocessor {
    /// Creates a new `ImagePreprocessor`.
    pub fn new(
        height: u32,
        width: u32,
        mean: Vec<f32>,
        std: Vec<f32>,
        bgr: bool,
    ) -> Self {
        Self {
            height,
            width,
            mean,
            std,
            bgr,
        }
    }

    /// Creates a preprocessor from a pretrained model's configuration on the Hugging Face Hub.
    pub async fn from_pretrained(repo_id: &str) -> Result<Self> {
        if let Ok(config) = PreprocessConfig::from_pretrained(repo_id).await {
            Self::from_preprocess_config(config)
        } else {
            Self::from_model_config(repo_id).await
        }
    }

    /// Creates a preprocessor from a `PreprocessConfig`.
    fn from_preprocess_config(config: PreprocessConfig) -> Result<Self> {
        let (height, width) = config
            .stages
            .iter()
            .find_map(|s| {
                if s.stage_type == "resize" {
                    s.size
                        .as_ref()
                        .and_then(|sz| (sz.len() == 2).then_some((sz[0], sz[1])))
                } else {
                    None
                }
            })
            .context("Resize configuration not found")?;

        let (mean, std) = config
            .stages
            .iter()
            .find_map(|s| {
                if s.stage_type == "normalize" {
                    Some((
                        s.mean.clone().unwrap_or_else(|| vec![0.5, 0.5, 0.5]),
                        s.std.clone().unwrap_or_else(|| vec![0.5, 0.5, 0.5]),
                    ))
                } else {
                    None
                }
            })
            .unwrap_or((vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]));

        Ok(Self::new(height, width, mean, std, false))
    }

    /// Creates a preprocessor from a `ModelConfig` as a fallback.
    async fn from_model_config(repo_id: &str) -> Result<Self> {
        let model_config = ModelConfig::from_pretrained(repo_id).await?;
        let input_size = &model_config.pretrained_cfg.input_size;
        anyhow::ensure!(input_size.len() == 3, "Invalid input size");

        let mean = vec![0.48145466, 0.4578275, 0.40821073];
        let std = vec![0.26862954, 0.26130258, 0.27577711];

        Ok(Self::new(
            input_size[1],
            input_size[2],
            mean,
            std,
            true,
        ))
    }

    /// Normalizes the pixel values and arranges them in the required tensor format.
    fn normalize_and_to_tensor(&self, image: &RgbImage) -> Array<f32, Ix4> {
        // Convert the image to an ndarray Array3
        let image_data = image.as_raw();
        let array = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 3),
            image_data.iter().map(|&x| x as f32 / 255.0).collect(),
        )
        .expect("Failed to create ndarray from image");

        // Create mean and std arrays for broadcasting
        let mean = Array3::from_shape_vec(
            (1, 1, 3),
            self.mean.clone(),
        )
        .expect("Failed to create mean array");

        let std = Array3::from_shape_vec(
            (1, 1, 3),
            self.std.clone(),
        )
        .expect("Failed to create std array");

        // Normalize the array using broadcasting
        let mut normalized_array = (array - mean) / std;

        // Permute the axes if needed (from HWC to CHW for NCHW layout)
        if !self.bgr {
            normalized_array = normalized_array.permuted_axes([2, 0, 1]).to_owned();
        }

        // Add the batch dimension (N)
        normalized_array.insert_axis(Axis(0))
    }
}

impl ImageProcessor for ImagePreprocessor {
    /// Preprocesses the image for model input by handling transparency, padding, resizing, and normalization.
    fn process(&self, image: &DynamicImage) -> Result<Array<f32, Ix4>> {
        let thumbnail = image.thumbnail(self.width, self.height);
        let thumbnail_rgb = thumbnail.to_rgb8();
        let (thumb_width, thumb_height) = thumbnail_rgb.dimensions();

        let mut padded_image =
            RgbImage::from_pixel(self.width, self.height, Rgb([128, 128, 128]));

        let pad_left = (self.width - thumb_width) / 2;
        let pad_top = (self.height - thumb_height) / 2;
        image::imageops::overlay(
            &mut padded_image,
            &thumbnail_rgb,
            pad_left as i64,
            pad_top as i64,
        );

        Ok(self.normalize_and_to_tensor(&padded_image))
    }
}
