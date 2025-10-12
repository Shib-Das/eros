//! # Eros
//!
//! A comprehensive, terminal-based application for processing and tagging media files.
//!
//! This application provides a TUI for selecting directories, configuring processing options,
//! and monitoring the progress of media file processing. It uses the `eros` library to
//! perform the actual tagging and optimization of images and videos.

mod app;
mod args;
mod ascii;
mod db;
mod file;
mod tag;
mod tui;
mod ui;
mod video;

use anyhow::Result;
use app::App;
use ffmpeg_next as ffmpeg;

/// The main entry point for the `eros` TUI application.
///
/// This function initializes the application, sets up the terminal, runs the main application loop,
/// and restores the terminal to its original state upon completion.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the `ffmpeg` library.
    ffmpeg::init()?;

    // Set up the terminal for the TUI.
    let mut terminal = tui::setup_terminal()?;

    // Create a new `App` instance and run the application.
    let mut app = App::default();
    app.run(&mut terminal).await?;

    // Restore the terminal to its original state.
    tui::restore_terminal(&mut terminal)?;

    Ok(())
}