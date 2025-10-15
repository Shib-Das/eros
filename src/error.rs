use thiserror::Error;

#[derive(Error, Debug)]
pub enum ErosError {
    #[error("I/O error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Optimizer error: {0}")]
    Optimizer(String),

    #[error("FFmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, ErosError>;