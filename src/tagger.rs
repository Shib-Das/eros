//! This module provides the `TaggerModel` struct for running ONNX-based image tagging models.
//!
//! It includes functionality for:
//! - Loading models from local paths or Hugging Face repositories.
//! - Selecting execution providers (e.g., CPU, CUDA).
//! - Running predictions on preprocessed image tensors.
//!
//! The `Device` enum allows for specifying the hardware to run the model on,
//! and the `TaggerModel` handles the ONNX Runtime session and prediction logic.

use std::path::Path;

use anyhow::Result;
use ndarray::{Array, Axis, Ix4};
use num_cpus;
use ort::{session::Session, value::Tensor, execution_providers::CPUExecutionProvider};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;

use crate::{error::TaggerError, file::TaggerModelFile};

/// Represents the execution device for the ONNX model.
///
/// This enum allows specifying which hardware to use for inference,
/// such as CPU, CUDA, or TensorRT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// Use the CPU for inference.
    Cpu,
    /// Use the CUDA execution provider.
    #[cfg(feature = "cuda")]
    Cuda(i32),
    /// Use the TensorRT execution provider.
    #[cfg(feature = "tensorrt")]
    TensorRT(i32),
    /// Use the CoreML execution provider (for macOS).
    #[cfg(feature = "coreml")]
    CoreML,
}

impl Device {
    /// Creates a list of `Device` instances for CPU execution.
    pub fn cpu() -> Vec<Self> {
        vec![Self::Cpu]
    }

    /// Creates a list of `Device` instances for CUDA execution on specified GPUs.
    #[cfg(feature = "cuda")]
    pub fn cuda_devices(device_ids: Vec<i32>) -> Vec<Self> {
        device_ids.into_iter().map(Self::Cuda).collect()
    }

    /// Creates a list of `Device` instances for TensorRT execution on specified GPUs.
    #[cfg(feature = "tensorrt")]
    pub fn tensorrt_devices(device_ids: Vec<i32>) -> Vec<Self> {
        device_ids.into_iter().map(Self::TensorRT).collect()
    }

    /// Creates a list of `Device` instances for CoreML execution.
    #[cfg(feature = "coreml")]
    pub fn coreml() -> Vec<Self> {
        vec![Self::CoreML]
    }
}

/// A wrapper around an ONNX Runtime session for image tagging.
///
/// This struct handles loading the model, managing the session, and running predictions.
#[derive(Debug)]
pub struct TaggerModel {
    session: Session,
    output_name: String,
}

impl TaggerModel {
    /// Initializes the ONNX Runtime with a list of execution providers.
    ///
    /// This function should be called once before creating any `TaggerModel` instances.
    /// It configures the global ONNX Runtime environment with the specified devices.
    pub fn init(devices: Vec<Device>) -> Result<(), TaggerError> {
        // Suppress verbose logging from ONNX Runtime
        let _ = tracing_subscriber::fmt::try_init();

        let mut providers = Vec::new();
        for device in devices {
            let provider = match device {
                Device::Cpu => CPUExecutionProvider::default().build(),
                #[cfg(feature = "cuda")]
                Device::Cuda(device_id) => CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .with_unified_memory(true)
                    .build(),
                #[cfg(feature = "tensorrt")]
                Device::TensorRT(device_id) => TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
                #[cfg(feature = "coreml")]
                Device::CoreML => CoreMLExecutionProvider::default().build(),
            };
            providers.push(provider);
        }

        ort::init()
            .with_execution_providers(providers)
            .commit()
            .map_err(|e| TaggerError::Ort(e.to_string()))?;
        Ok(())
    }

    /// Loads a model from a local file path.
    ///
    /// The path should point to a valid `.onnx` model file.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, TaggerError> {
        let threads = num_cpus::get();
        let session = Session::builder()
            .and_then(|b| b.with_parallel_execution(true))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.with_intra_threads(threads))
            .and_then(|b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| TaggerError::Ort(e.to_string()))?;

        let output_name = session
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| TaggerError::Ort("Model has no outputs".to_string()))?;

        Ok(Self {
            session,
            output_name,
        })
    }

    /// Loads a model from a Hugging Face repository.
    ///
    /// This will download the model file if it's not already cached.
    pub async fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let model_path = TaggerModelFile::new(repo_id).get().await?;
        Self::load(&model_path)
    }

    /// Runs prediction on a batch of preprocessed image tensors.
    ///
    /// # Arguments
    ///
    /// * `input_tensor` - A 4D tensor with shape `[batch_size, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// A nested vector where each inner vector contains the prediction probabilities for one image.
    pub fn predict(&mut self, input_tensor: Array<f32, Ix4>) -> Result<Vec<Vec<f32>>, TaggerError> {
        let input_tensor =
            Tensor::from_array(input_tensor).map_err(|e| TaggerError::Ort(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input_tensor])
            .map_err(|e| TaggerError::Ort(e.to_string()))?;

        let preds = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| TaggerError::Ort(e.to_string()))?;

        let preds_vec = preds
            .axis_iter(Axis(0))
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect();

        Ok(preds_vec)
    }
}

