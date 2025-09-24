mod app;
mod args;
mod db;
mod file;
mod tag;
mod ui;
mod video;

use anyhow::Result;
use app::{App, AppConfig};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use db::Database;
use eros::{
    file::{ConfigFile, HfFile, TagCSVFile, TaggerModelFile},
    pipeline::TaggingPipeline,
    processor::ImagePreprocessor,
    tagger::{Device, TaggerModel},
    tags::LabelTags,
};
use futures::{StreamExt, TryStreamExt};
use futures_batch::ChunksTimeoutStreamExt;
use hf_hub::Cache;
use ratatui::{backend::CrosstermBackend, Terminal};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::file::TaggingResultSimple;

fn get_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 1024];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run the menu
    let mut app = App::default();
    let config = app.run(&mut terminal)?;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // If config is Some, user wants to proceed
    if let Some(config) = config {
        run_tagging_process(config).await?;
    }

    Ok(())
}

async fn run_tagging_process(config: AppConfig) -> Result<()> {
    println!("Starting tagging process with config: {:?}", config);

    let device = Device::cpu();
    let repo_id = config.model.repo_id();

    let models_dir = PathBuf::from("./models");
    fs::create_dir_all(&models_dir)?;
    let cache = Cache::new(models_dir);

    println!("Downloading model files to ./models/...");
    let model_file_path = TaggerModelFile::new(&repo_id).get_with_cache(cache.clone())?;
    let config_file_path = ConfigFile::new(&repo_id).get_with_cache(cache.clone())?;
    let tag_csv_file_path = TagCSVFile::new(&repo_id).get_with_cache(cache.clone())?;
    println!("Download complete.");

    TaggerModel::use_devices(device)?;
    let model = TaggerModel::load(&model_file_path)?;
    let preprocessor =
        ImagePreprocessor::from_config(&eros::config::ModelConfig::load(&config_file_path)?)?;
    let label_tags = LabelTags::load(&tag_csv_file_path)?;

    let pipe = TaggingPipeline::new(model, preprocessor, label_tags, &config.threshold);
    let pipe = Arc::new(Mutex::new(pipe));

    fs::create_dir_all("./data")?;
    let db = Database::new("./data/victim.db")?;
    db.init()?;
    let db = Arc::new(Mutex::new(db));

    // --- Image Processing ---
    let image_path = PathBuf::from_str(&config.input_path)?;
    if file::is_file(&image_path).await? {
        println!("Processing single image file...");
        let img = image::open(&image_path)?;
        let result = pipe.lock().unwrap().predict(img)?;
        let simple_result = TaggingResultSimple::from(result);
        let hash = get_hash(&image_path)?;
        let size = fs::metadata(&image_path)?.len();
        db.lock().unwrap().save_image_tags(
            image_path.to_str().unwrap(),
            size,
            &hash,
            &simple_result.tags,
        )?;
    } else {
        let image_files = file::get_image_files(image_path.to_str().unwrap()).await?;
        if !image_files.is_empty() {
            println!("Found {} image files. Starting processing.", image_files.len());
            // This section can be expanded with the TUI for progress if desired
        }
    }

    // --- Video Processing ---
    let video_path = PathBuf::from_str(&config.video_path)?;
    if file::is_file(&video_path).await? {
        video::process_video(&video_path, &pipe, &db, get_hash).await?;
    } else {
        let video_files = video::get_video_files(video_path.to_str().unwrap()).await?;
        for video_file in video_files {
            video::process_video(&video_file, &pipe, &db, get_hash).await?;
        }
    }

    println!("\nAll processing complete.");
    Ok(())
}