use thiserror::Error;

/// Represents the possible errors that can occur within the tagging library.
#[derive(Error, Debug)]
pub enum TaggerError {
    /// An error occurred while interacting with the Hugging Face API.
    #[error("Hugging Face API error: {0}")]
    Hf(String),
    /// An error occurred within the ONNX Runtime.
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
    /// An error related to CUDA occurred.
    #[error("CUDA error: {0}")]
    Cuda(String),
    /// An error occurred during image processing.
    #[error("Processor error: {0}")]
    Processor(String),
    /// An error related to tag processing or loading occurred.
    #[error("Tag error: {0}")]
    Tag(String),
    /// An error occurred within the tagging pipeline.
    #[error("Pipeline error: {0}")]
    Pipeline(String),
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(String),
    /// A network-related error occurred.
    #[error("Network error: {0}")]
    Network(String),
    /// An error occurred during JSON serialization or deserialization.
    #[error("JSON error: {0}")]
    Json(String),
    /// An error occurred during content rating.
    #[error("Rating error: {0}")]
    Rating(String),
}

impl From<std::io::Error> for TaggerError {
    fn from(err: std::io::Error) -> Self {
        TaggerError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for TaggerError {
    fn from(err: serde_json::Error) -> Self {
        TaggerError::Json(err.to_string())
    }
}

impl From<ort::Error> for TaggerError {
    fn from(err: ort::Error) -> Self {
        TaggerError::Ort(err.to_string())
    }
}