//! # Eros
//!
//! Eros is a library for image tagging using ONNX models. It provides a flexible
//! and efficient pipeline for processing images, running inference, and generating
//! descriptive tags.
//!
//! ## Features
//!
//! - **High-level API**: A simple `TaggingPipeline` for end-to-end image tagging.
//! - **ONNX Runtime**: Powered by `ort` for efficient, cross-platform inference.
//! - **Execution Providers**: Supports CPU, CUDA, and other execution providers.
//! - **Preprocessing**: Includes tools for resizing, padding, and normalizing images.
//! - **Extensible**: Designed with traits to allow for custom components.
//!
//! ## Modules
//!
//! - `pipeline`: The main entry point for using the tagging functionality.
//! - `tagger`: Handles the ONNX model and session management.
//! - `processor`: Provides tools for image preprocessing.
//! - `tags`: Manages tag labels and their categories.
//! - `config`: Defines the data structures for model configuration.
//! - `error`: Contains the error types for the library.
//! - `prelude`: A collection of the most commonly used types.

pub mod config;
pub mod error;
pub mod file;
pub mod pipeline;
pub mod prelude;

pub mod optimizer;
pub mod processor;
pub mod rating;
pub mod tagger;
pub mod tags;
