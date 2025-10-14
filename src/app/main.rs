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
mod core;
mod db;
mod file;
mod tag;
mod tui;
mod ui;
mod video;

use anyhow::Result;
use app::{App, ProgressUpdate};
use args::{Args, Commands, V3Model};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// The main entry point for the `eros` application.
///
/// This function initializes the application, parses command-line arguments, and
/// launches either the TUI or the CLI mode.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the `ffmpeg` library.
    ffmpeg::init()?;

    let args = Args::parse();

    match args.command {
        Some(Commands::Process { path, threshold }) => {
            run_cli(path, threshold).await?;
        }
        None => {
            run_tui().await?;
        }
    }

    Ok(())
}

/// Runs the application in CLI mode.
async fn run_cli(path: String, threshold: f32) -> Result<()> {
    let (tx, mut rx) = mpsc::channel(100);

    let config = core::AppConfig {
        model: V3Model::SwinV2,
        input_path: path.clone(),
        video_path: path.clone(),
        threshold,
        batch_size: 1,
        show_ascii_art: false,
    };
    let selected_dirs = vec![PathBuf::from(path)];

    // Spawn the processing task
    tokio::spawn(async move {
        if let Err(e) = core::run_full_process(config, selected_dirs, tx.clone()).await {
            let _ = tx.send(ProgressUpdate::Error(e.to_string())).await;
        }
    });

    // Handle progress updates
    while let Some(update) = rx.recv().await {
        match update {
            ProgressUpdate::Message(msg) => println!("{}", msg),
            ProgressUpdate::Progress(p) => {
                println!("Progress: {:.2}%", p * 100.0);
            }
            ProgressUpdate::Error(e) => {
                eprintln!("Error: {}", e);
                break;
            }
            ProgressUpdate::Complete => {
                println!("Processing complete!");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Runs the application in TUI mode.
async fn run_tui() -> Result<()> {
    // Set up the terminal for the TUI.
    let mut terminal = tui::setup_terminal()?;

    // Create a new `App` instance and run the application.
    let mut app = App::default();
    app.run(&mut terminal).await?;

    // Restore the terminal to its original state.
    tui::restore_terminal(&mut terminal)?;

    Ok(())
}