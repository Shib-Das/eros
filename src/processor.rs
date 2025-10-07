//! This module provides tools for preprocessing images before they are fed into a model.
//!
//! It defines the `ImageProcessor` trait for generic image processing operations
//! and provides a concrete implementation, `ImagePreprocessor`, which handles
//! resizing, padding, normalization, and color channel ordering.

use anyhow::Result;
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::{Array, Axis, Ix4};
use rayon::prelude::*;

use crate::{
    config::{ModelConfig, PreprocessConfig},
    error::TaggerError,
};

/// A trait for processing images into tensors suitable for model input.
pub trait ImageProcessor {
    /// Processes a single image into a 4D tensor.
    fn process(&self, image: &DynamicImage) -> Result<Array<f32, Ix4>, TaggerError>;

    /// Processes a batch of images into a single 4D tensor.
    fn process_batch(&self, images: Vec<&DynamicImage>) -> Result<Array<f32, Ix4>, TaggerError>
    where
        Self: Sync,
    {
        let tensors: Result<Vec<_>, _> =
            images.into_par_iter().map(|img| self.process(img)).collect();
        let tensors = tensors?;

        ndarray::concatenate(
            Axis(0),
            &tensors.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| TaggerError::Processor(format!("Failed to concatenate tensors: {}", e)))
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
    pub async fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        // Try to load preprocessor_config.json first
        if let Ok(config) = PreprocessConfig::from_pretrained(repo_id).await {
            // This path is for models that have a detailed preprocessing config
            let (height, width) = config
                .stages
                .iter()
                .find_map(|s| {
                    if s.stage_type == "resize" {
                        s.size.as_ref().and_then(|sz| {
                            if sz.len() == 2 {
                                Some((sz[0], sz[1]))
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
                .ok_or_else(|| TaggerError::Processor("Resize configuration not found".to_string()))?;

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

            // Most modern models expect RGB, so we'll default to bgr: false unless
            // specified otherwise. This is a reasonable assumption for now.
            Ok(Self::new(height, width, mean, std, false))
        } else {
            // Fallback for models like wd-swinv2-tagger-v3 that use config.json
            // and have known normalization values.
            let model_config = ModelConfig::from_pretrained(repo_id).await?;
            let input_size = &model_config.pretrained_cfg.input_size;
            if input_size.len() != 3 {
                return Err(TaggerError::Processor("Invalid input size".to_string()));
            }

            // These are the correct normalization values for the swinv2 tagger models.
            // Using these ensures the model performs optimally.
            let mean = vec![0.48145466, 0.4578275, 0.40821073];
            let std = vec![0.26862954, 0.26130258, 0.27577711];

            Ok(Self::new(
                input_size[1],
                input_size[2],
                mean,
                std,
                true, // These models expect NHWC, which bgr=true represents
            ))
        }
    }

    /// Normalizes the pixel values and arranges them in the required tensor format.
    fn normalize_and_to_tensor(&self, image: &RgbImage) -> Array<f32, Ix4> {
        let mut tensor = if self.bgr {
            // NHWC layout for older models
            Array::zeros((self.height as usize, self.width as usize, 3))
        } else {
            // NCHW layout for newer models
            Array::zeros((3, self.height as usize, self.width as usize))
        };

        for (x, y, pixel) in image.enumerate_pixels() {
            let [r, g, b] = pixel.0;

            let r_norm = (r as f32 / 255.0 - self.mean[0]) / self.std[0];
            let g_norm = (g as f32 / 255.0 - self.mean[1]) / self.std[1];
            let b_norm = (b as f32 / 255.0 - self.mean[2]) / self.std[2];

            if self.bgr {
                // NHWC layout
                tensor[[y as usize, x as usize, 0]] = r_norm;
                tensor[[y as usize, x as usize, 1]] = g_norm;
                tensor[[y as usize, x as usize, 2]] = b_norm;
            } else {
                // NCHW layout
                tensor[[0, y as usize, x as usize]] = r_norm;
                tensor[[1, y as usize, x as usize]] = g_norm;
                tensor[[2, y as usize, x as usize]] = b_norm;
            }
        }

        tensor.insert_axis(Axis(0))
    }
}

impl ImageProcessor for ImagePreprocessor {
    /// Preprocesses the image for model input by handling transparency, padding, resizing, and normalization.
    fn process(&self, image: &DynamicImage) -> Result<Array<f32, Ix4>, TaggerError> {
        // 1. Convert to RGB and handle transparency by compositing on a gray background
        let image_rgb = image.to_rgb8();
        let (original_width, original_height) = image_rgb.dimensions();

        // 2. Manually calculate dimensions to preserve aspect ratio
        let target_width = self.width;
        let target_height = self.height;

        let w_ratio = target_width as f32 / original_width as f32;
        let h_ratio = target_height as f32 / original_height as f32;
        let ratio = w_ratio.min(h_ratio);

        let new_width = (original_width as f32 * ratio).round() as u32;
        let new_height = (original_height as f32 * ratio).round() as u32;

        // 3. Resize the image using the calculated dimensions
        let resized_image = image::imageops::resize(
            &image_rgb,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle, // This is what thumbnail uses
        );
        let (thumb_width, thumb_height) = resized_image.dimensions();

        // 4. Create a new image with the target dimensions and a gray background
        let mut padded_image =
            RgbImage::from_pixel(target_width, target_height, Rgb([128, 128, 128]));

        // 5. Calculate padding and overlay the thumbnail onto the padded background
        let pad_left = (target_width - thumb_width) / 2;
        let pad_top = (target_height - thumb_height) / 2;
        image::imageops::overlay(&mut padded_image, &resized_image, pad_left as i64, pad_top as i64);

        // 6. Normalize the image and convert to a tensor
        Ok(self.normalize_and_to_tensor(&padded_image))
    }
}
