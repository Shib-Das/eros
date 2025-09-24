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

    // load model
    TaggerModel::use_devices(device)?; // do once
    let model = TaggerModel::load(&model_file_path)?;
    let model_config = ModelConfig::load(&config_file_path)?;
    let preprocessor = ImagePreprocessor::from_config(&model_config)?;
    let label_tags = LabelTags::load(&tag_csv_file_path)?;

    // I/O
    let input = PathBuf::from_str(&config.input_path)?;

    // load pipe
    let mut pipe = TaggingPipeline::new(model, preprocessor, label_tags, &config.threshold);

    // database
    fs::create_dir_all("./data")?;
    let db = Database::new("./data/victim.db")?;
    db.init()?;
    let db = Arc::new(Mutex::new(db));

    if file::is_file(&input).await? {
        if file::is_video(input.to_str().unwrap())? {
            println!("Processing single video file...");
            process_video(&input, &mut pipe, &db).await?;
        } else {
            println!("Processing single image file...");
            let img = image::open(&input)?;
            let result = pipe.predict(img)?;
            let simple_result = TaggingResultSimple::from(result);
            let hash = get_hash(&input)?;
            let size = fs::metadata(&input)?.len();

            db.lock().unwrap().save_image_tags(
                input.to_str().unwrap(),
                size,
                &hash,
                &simple_result.tags,
            )?;
            println!("Done.");
        }
    } else {
        let image_files = file::get_image_files(input.to_str().unwrap()).await?;
        let total_images = image_files.len() as u64;
        println!("Found {} image files. Starting TUI progress view.", total_images);

        let progress = Arc::new(Mutex::new(0u64));
        let tui_progress = progress.clone();

        let tui_thread = std::thread::spawn(move || -> Result<()> {
            enable_raw_mode()?;
            let mut stdout = io::stdout();
            execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
            let backend = CrosstermBackend::new(stdout);
            let mut terminal = Terminal::new(backend)?;

            loop {
                let current = *tui_progress.lock().unwrap();
                terminal.draw(|f| ui::draw_progress(f, total_images, current, "", ""))?;
                if current >= total_images {
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

                async move {
                    let imgs = image_paths
                        .iter()
                        .map(|path| {
                            image::open(path)
                                .map_err(|e| anyhow::anyhow!("Failed to open image {}: {}", path.display(), e))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let results =
                        tokio::task::block_in_place(|| pipe.lock().unwrap().predict_batch(imgs.iter().collect()))?;

                    for (image_path, result) in image_paths.iter().zip(results.iter()) {
                        let simple_result = TaggingResultSimple::from(result.clone());
                        let hash = get_hash(image_path)?;
                        let size = fs::metadata(image_path)?.len();

                        db.lock().unwrap().save_image_tags(
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

        tui_thread.join().unwrap()?;
        println!("Processing complete.");
    }
    Ok(())
}
async fn process_video(
    video_path: &Path,
    pipe: &mut TaggingPipeline,
    db: &Arc<Mutex<Database>>,
) -> Result<()> {
    ffmpeg_next::init().unwrap();
    let mut ictx = ffmpeg_next::format::input(&video_path)?;
    let input = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or(ffmpeg_next::Error::StreamNotFound)?;
    let video_stream_index = input.index();

    let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg_next::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
    )?;

    let mut all_tags = HashSet::new();
    let mut frame_count = 0;
    let frame_rate = input.avg_frame_rate();
    let frame_rate_f64 = frame_rate.0 as f64 / frame_rate.1 as f64;


    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if frame_count % (5.0 * frame_rate_f64) as i32 == 0 {
                    let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;

                    let img = image::DynamicImage::ImageRgb8(
                        image::RgbImage::from_raw(
                            rgb_frame.width(),
                            rgb_frame.height(),
                            rgb_frame.data(0).to_vec(),
                        )
                        .unwrap(),
                    );

                    let result = pipe.predict(img)?;
                    let simple_result = TaggingResultSimple::from(result);
                    simple_result
                        .tags
                        .split(", ")
                        .for_each(|tag| {
                            all_tags.insert(tag.to_string());
                        });
                }
                frame_count += 1;
            }
        }
    }

    let tags = all_tags.into_iter().collect::<Vec<String>>().join(", ");
    let hash = get_hash(video_path)?;
    let size = fs::metadata(video_path)?.len();
    db.lock().unwrap().save_video_tags(
        video_path.to_str().unwrap(),
        size,
        &hash,
        &tags,
    )?;
    println!("Video processing complete.");
    Ok(())
}