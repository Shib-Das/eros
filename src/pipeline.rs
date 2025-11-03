//! This module provides a high-level `TaggingPipeline` for processing images and generating tags.
//!
//! The pipeline combines a `TaggerModel` and an `ImagePreprocessor` to create an
//! end-to-end solution for image tagging. It handles model loading, image preprocessing,
//! prediction, and post-processing of the results into categorized and sorted tags.
//!
//! The main components are `TaggingPipeline` for managing the workflow and `TaggingResult`
//! for representing the output.

use anyhow::{Context, Result};
use image::DynamicImage;
use indexmap::IndexMap;
use itertools::Itertools;

use crate::{
    processor::{ImagePreprocessor, ImageProcessor},
    tagger::{Device, TaggerModel},
    tags::{LabelTags, TagCategory},
};

/// A callback function for reporting progress.
///
/// The first argument is the progress percentage (0.0 to 1.0), and the second
/// is a status message.
pub type ProgressCallback = Box<dyn Fn(f32, String) + Send + Sync>;

/// An end-to-end pipeline for image tagging.
#[derive(Debug)]
pub struct TaggingPipeline {
    /// The underlying ONNX model for tagging.
    pub model: TaggerModel,
    /// The preprocessor for preparing images.
    pub preprocessor: ImagePreprocessor,
    /// The set of labels the model can predict.
    pub tags: LabelTags,
    /// The confidence threshold for including a tag in the results.
    pub threshold: f32,
}

/// A type alias for a map of tag predictions, from tag name to confidence score.
pub type Prediction = IndexMap<String, f32>;

/// The result of a tagging operation, with tags categorized and sorted by confidence.
#[derive(Debug, Clone)]
pub struct TaggingResult {
    /// Rating tags (e.g., "safe", "sensitive").
    pub rating: Prediction,
    /// Character tags.
    pub character: Prediction,
    /// General-purpose tags.
    pub general: Prediction,
}

impl TaggingResult {
    /// Creates a new `TaggingResult` from categorized predictions.
    fn new(rating: Prediction, character: Prediction, general: Prediction) -> Self {
        Self {
            rating,
            character,
            general,
        }
    }
}

impl TaggingPipeline {
    /// Creates a new `TaggingPipeline`.
    pub fn new(
        model: TaggerModel,
        preprocessor: ImagePreprocessor,
        tags: LabelTags,
        threshold: &f32,
    ) -> Self {
        Self {
            model,
            preprocessor,
            tags,
            threshold: *threshold,
        }
    }

    /// Creates a new `TaggingPipeline` from a pretrained model on the Hugging Face Hub.
    pub async fn from_pretrained(
        model_name: &str,
        devices: Vec<Device>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Self> {
        let progress_callback = progress_callback.as_ref();

        Self::report_progress(progress_callback, 0.0, "Initializing Tagger...");
        TaggerModel::init(devices)?;

        Self::report_progress(
            progress_callback,
            0.2,
            &format!("Downloading model: {}", model_name),
        );
        let model = TaggerModel::from_pretrained(model_name).await?;

        Self::report_progress(progress_callback, 0.5, "Setting up preprocessor...");
        let preprocessor = ImagePreprocessor::from_pretrained(model_name).await?;

        Self::report_progress(progress_callback, 0.8, "Downloading tags...");
        let tags = LabelTags::from_pretrained(model_name).await?;

        Self::report_progress(progress_callback, 1.0, "Pipeline ready.");

        Ok(Self {
            model,
            preprocessor,
            tags,
            threshold: 0.5,
        })
    }

    /// Reports progress using the provided callback.
    fn report_progress(
        progress_callback: Option<&ProgressCallback>,
        progress: f32,
        message: &str,
    ) {
        if let Some(cb) = progress_callback {
            cb(progress, message.to_string());
        }
    }

    /// Filters and sorts tags for a specific category from a set of predictions.
    fn get_tags_for_category(&self, pairs: &Prediction, category: TagCategory) -> Prediction {
        pairs
            .iter()
            .filter(|(tag, &prob)| {
                prob >= self.threshold
                    && self
                        .tags
                        .label2tag()
                        .get(*tag)
                        .map_or(false, |t| t.category() == category)
            })
            .sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tag, &prob)| (tag.clone(), prob))
            .collect()
    }

    /// Predicts tags for a single image.
    pub fn predict(
        &mut self,
        image: DynamicImage,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<TaggingResult> {
        let mut results = self.predict_batch(vec![&image], progress_callback)?;
        results
            .pop()
            .context("Prediction batch returned no results for a single image")
    }

    /// Predicts tags for a batch of images.
    pub fn predict_batch(
        &mut self,
        images: Vec<&DynamicImage>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<TaggingResult>> {
        let progress_callback = progress_callback.as_ref();
        Self::report_progress(progress_callback, 0.0, "Preprocessing images...");
        let tensor = self.preprocessor.process_batch(images)?;

        Self::report_progress(progress_callback, 0.3, "Running model prediction...");
        let probs = self.model.predict(tensor)?;

        Self::report_progress(progress_callback, 0.6, "Processing results...");
        let pairs_batch = self.tags.create_probality_pairs(probs)?;

        let results = pairs_batch
            .iter()
            .map(|pairs| {
                let rating = self.get_tags_for_category(pairs, TagCategory::Rating);
                let character = self.get_tags_for_category(pairs, TagCategory::Character);
                let general = self.get_tags_for_category(pairs, TagCategory::General);
                TaggingResult::new(rating, character, general)
            })
            .collect();

        Self::report_progress(progress_callback, 1.0, "Prediction complete.");

        Ok(results)
    }
}

