//! # Error Handling
//!
//! This module defines the custom error type for the `eros` library.
//!
//! The `TaggerError` enum represents all possible errors that can occur
//! within the library, providing a unified and consistent error-handling mechanism.
//! It uses the `thiserror` crate to derive the `Error` trait and provide
//! descriptive error messages.

pub type TaggerError = anyhow::Error;