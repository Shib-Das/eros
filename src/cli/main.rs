mod app;
mod args;
mod db;
mod file;
mod tag;
mod ui;

use anyhow::Result;
use app::{App, AppConfig};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use db::Database;
use eros::{
    config::ModelConfig,
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
    collections::HashSet,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};
use sysinfo::System;
use ffmpeg_sidecar::{command::FfmpegCommand, event::FfmpegEvent};


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

    // Create a local cache for models
    let models_dir = PathBuf::from("./models");
    fs::create_dir_all(&models_dir)?;
    let cache = Cache::new(models_dir);

    // define files
    let model_file = TaggerModelFile::new(&repo_id);
    let config_file = ConfigFile::new(&repo_id);
    let tag_csv_file = TagCSVFile::new(&repo_id);

    // pre-download files to the local cache
    println!("Downloading model files to ./models/...");
    let model_file_path = model_file.get_with_cache(cache.clone())?;
    let config_file_path = config_file.get_with_cache(cache.clone())?;
    let tag_csv_file_path = tag_csv_file.get_with_cache(cache.clone())?;
    println!("Download complete.");
    
    // download ffmpeg
    println!("Downloading ffmpeg...");
    ffmpeg_sidecar::download::auto_download().unwrap();
    println!("Download complete.");

    // load model
    TaggerModel::use_devices(device)?; // do once
    let model = TaggerModel::load(&model_file_path)?;
    let model_config = ModelConfig::load(&config_file_path)?;
    let preprocessor = ImagePreprocessor::from_config(&model_config)?;
    let label_tags = LabelTags::load(&tag_csv_file_path)?;

    // I/O
    let input = PathBuf::from_str(&config.input_path)?;
    let video_input = PathBuf::from_str(&config.video_path)?;

    // load pipe
    let mut pipe = TaggingPipeline::new(model, preprocessor, label_tags, &config.threshold);

    // database
    fs::create_dir_all("./data")?;
    let db = Database::new("./data/victim.db")?;
    db.init()?;
    let db = Arc::new(Mutex::new(db));

    match file::is_file(&input).await? {
        true => {
            println!("Processing single file...");
            let img = image::open(&input)?;
            let result = pipe.predict(img)?;
            let simple_result = TaggingResultSimple::from(result);
            let hash = get_hash(&input)?;
            let size = fs::metadata(&input)?.len();

            db.lock().unwrap().save_tags(
                input.to_str().unwrap(),
                size,
                &hash,
                &simple_result.tags,
            )?;
            println!("Done.");
        }
        false => {
            let image_files = file::get_image_files(input.to_str().unwrap()).await?;
            let video_files = file::get_video_files(video_input.to_str().unwrap()).await?;
            let total_images = image_files.len() as u64;
            let total_videos = video_files.len() as u64;

            println!(
                "Found {} image files and {} video files. Starting TUI progress view.",
                total_images, total_videos
            );

            let progress = Arc::new(Mutex::new(0u64));
            let current_file = Arc::new(Mutex::new(String::new()));
            let ram_usage = Arc::new(Mutex::new(String::new()));

            let tui_progress = progress.clone();
            let tui_current_file = current_file.clone();
            let tui_ram_usage = ram_usage.clone();

            let tui_thread = std::thread::spawn(move || -> Result<()> {
                let mut sys = System::new_all();
                enable_raw_mode()?;
                let mut stdout = io::stdout();
                execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
                let backend = CrosstermBackend::new(stdout);
                let mut terminal = Terminal::new(backend)?;

                loop {
                    sys.refresh_all();
                    let current = *tui_progress.lock().unwrap();
                    let file = tui_current_file.lock().unwrap().clone();
                    let ram = format!(
                        "RAM: {} / {} MB",
                        sys.used_memory() / 1024 / 1024,
                        sys.total_memory() / 1024 / 1024
                    );
                    *tui_ram_usage.lock().unwrap() = ram;

                    terminal.draw(|f| {
                        ui::draw_progress(
                            f,
                            total_images + total_videos,
                            current,
                            &file,
                            &tui_ram_usage.lock().unwrap(),
                        )
                    })?;
                    if current >= total_images + total_videos {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }

                disable_raw_mode()?;
                execute!(
                    terminal.backend_mut(),
                    LeaveAlternateScreen,
                    DisableMouseCapture
                )?;
                terminal.show_cursor()?;
                Ok(())
            });

            let pipe = Arc::new(Mutex::new(pipe));

            futures::stream::iter(image_files)
                .chunks_timeout(config.batch_size, Duration::from_millis(100))
                .map(|image_paths| {
                    let pipe = Arc::clone(&pipe);
                    let progress = Arc::clone(&progress);
                    let db = Arc::clone(&db);
                    let current_file = Arc::clone(&current_file);

                    async move {
                        for image_path in &image_paths {
                            *current_file.lock().unwrap() =
                                image_path.to_str().unwrap().to_string();
                        }
                        let imgs = image_paths
                            .iter()
                            .map(|path| {
                                image::open(path).map_err(|e| {
                                    anyhow::anyhow!("Failed to open image {}: {}", path.display(), e)
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        let results = tokio::task::block_in_place(|| {
                            pipe.lock().unwrap().predict_batch(imgs.iter().collect())
                        })?;

                        for (image_path, result) in image_paths.iter().zip(results.iter()) {
                            let simple_result = TaggingResultSimple::from(result.clone());
                            let hash = get_hash(image_path)?;
                            let size = fs::metadata(image_path)?.len();

                            db.lock().unwrap().save_tags(
                                image_path.to_str().unwrap(),
                                size,
                                &hash,
                                &simple_result.tags,
                            )?;
                        }

                        let mut num = progress.lock().unwrap();
                        *num += image_paths.len() as u64;

                        anyhow::Ok(())
                    }
                })
                .buffer_unordered(4)
                .try_collect::<Vec<_>>()
                .await?;

            for video_path in video_files {
                *current_file.lock().unwrap() = video_path.to_str().unwrap().to_string();
                let temp_dir =
                    format!("./temp_{}", video_path.file_stem().unwrap().to_str().unwrap());
                fs::create_dir_all(&temp_dir)?;

                let mut all_tags = HashSet::new();

                let mut child = FfmpegCommand::new()
                    .input(video_path.to_str().unwrap())
                    .args(["-vf", "fps=1/5"])
                    .args(["-f", "image2pipe", "-c:v", "png", "pipe:1"])
                    .spawn()?;
                
                let mut frame_count = 0;
                for event in child.iter().unwrap() {
                    if let FfmpegEvent::OutputFrame(frame) = event {
                        let frame_path = format!("{}/frame_{}.png", temp_dir, frame_count);
                        fs::write(&frame_path, &frame.data)?;
                        frame_count += 1;

                        let img = image::open(&frame_path)?;
                        let result = pipe.lock().unwrap().predict(img)?;
                        let simple_result = TaggingResultSimple::from(result);

                        for tag in simple_result.tags.split(", ") {
                            all_tags.insert(tag.to_string());
                        }
                    }
                }

                child.wait()?;

                let hash = get_hash(&video_path)?;
                let size = fs::metadata(&video_path)?.len();
                let tags = all_tags.into_iter().collect::<Vec<_>>().join(", ");

                db.lock().unwrap().save_tags(
                    video_path.to_str().unwrap(),
                    size,
                    &hash,
                    &tags,
                )?;

                fs::remove_dir_all(&temp_dir)?;

                let mut num = progress.lock().unwrap();
                *num += 1;
            }

            tui_thread.join().unwrap()?;
            println!("Processing complete.");
        }
    }
    Ok(())
}