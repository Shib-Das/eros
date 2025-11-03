use anyhow::Result;
use image::DynamicImage;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

use crate::{
    args::V3Model,
    db::Database,
    file::{self, TaggingResultSimple},
    video,
};
use eros::{
    pipeline::TaggingPipeline,
    prelude::{self},
    rating::RatingModel,
    tagger::Device,
};

use super::app::ProgressUpdate;

/// Runs the full media processing pipeline.
pub async fn run_full_process(
    config: AppConfig,
    selected_dirs: Vec<PathBuf>,
    tx: mpsc::Sender<ProgressUpdate>,
) -> Result<()> {
    prepare_media_files(&selected_dirs, &tx).await?;
    let (pipe, rating_model, db) = initialize_pipeline_and_db(&config, &tx).await?;
    process_images(
        &selected_dirs,
        &pipe,
        &rating_model,
        &db,
        &tx,
        config.show_ascii_art,
    )
    .await?;
    process_videos(
        &selected_dirs,
        &pipe,
        &rating_model,
        &db,
        &tx,
        config.show_ascii_art,
    )
    .await?;

    tx.send(ProgressUpdate::Message(
        "Optimizing media files...".to_string(),
    ))
    .await?;
    eros::optimizer::optimize_media_in_dirs(&selected_dirs).await?;
    tx.send(ProgressUpdate::Progress(0.99)).await?;

    tx.send(ProgressUpdate::Complete).await?;
    Ok(())
}

/// Prepares media files by renaming, converting, and resizing them.
async fn prepare_media_files(
    selected_dirs: &[PathBuf],
    tx: &mpsc::Sender<ProgressUpdate>,
) -> Result<()> {
    tx.send(ProgressUpdate::Message("Renaming files...".to_string()))
        .await?;
    prelude::rename_files_in_selected_dirs(selected_dirs)?;
    tx.send(ProgressUpdate::Progress(0.05)).await?;

    tx.send(ProgressUpdate::Message(
        "Converting files and stripping metadata...".to_string(),
    ))
    .await?;
    prelude::convert_and_strip_metadata(selected_dirs)?;
    tx.send(ProgressUpdate::Progress(0.1)).await?;

    tx.send(ProgressUpdate::Message("Resizing media...".to_string()))
        .await?;
    prelude::resize_media(selected_dirs, (448, 448))?;
    tx.send(ProgressUpdate::Progress(0.15)).await?;
    Ok(())
}

/// Initializes the tagging pipeline and the database.
async fn initialize_pipeline_and_db(
    config: &AppConfig,
    tx: &mpsc::Sender<ProgressUpdate>,
) -> Result<(
    Arc<Mutex<TaggingPipeline>>,
    Arc<Mutex<RatingModel>>,
    Arc<Mutex<Database>>,
)> {
    let tx_clone = tx.clone();
    let progress_callback = Box::new(move |progress: f32, message: String| {
        let _ = tx_clone.try_send(ProgressUpdate::Message(message));
        let _ = tx_clone.try_send(ProgressUpdate::Progress(0.15 + (progress as f64 * 0.1)));
    });

    let mut pipe = TaggingPipeline::from_pretrained(
        &config.model.repo_id(),
        Device::cpu(),
        Some(progress_callback),
    )
    .await?;
    pipe.threshold = config.threshold;
    let pipe = Arc::new(Mutex::new(pipe));

    let rating_model = RatingModel::new().await?;
    let rating_model = Arc::new(Mutex::new(rating_model));

    tx.send(ProgressUpdate::Progress(0.25)).await?;

    fs::create_dir_all("./data")?;
    let db = Database::new("./data/victim.db")?;
    db.init()?;
    Ok((pipe, rating_model, Arc::new(Mutex::new(db))))
}

/// Processes all image files in the selected directories.
async fn process_images(
    selected_dirs: &[PathBuf],
    pipe: &Arc<Mutex<TaggingPipeline>>,
    rating_model: &Arc<Mutex<RatingModel>>,
    db: &Arc<Mutex<Database>>,
    tx: &mpsc::Sender<ProgressUpdate>,
    show_ascii_art: bool,
) -> Result<()> {
    let mut image_files = Vec::new();
    for dir in selected_dirs {
        if let Some(dir_str) = dir.to_str() {
            image_files.extend(file::get_image_files(dir_str).await?);
        }
    }

    let total_images = image_files.len();
    if total_images > 0 {
        tx.send(ProgressUpdate::Message(format!(
            "Processing {} image files...",
            total_images
        )))
        .await?;
        for (i, image_file) in image_files.into_iter().enumerate() {
            let img = image::open(&image_file)?;
            if show_ascii_art {
                // We don't care if this fails, it just means the UI closed.
                let _ = tx
                    .send(ProgressUpdate::ImageProcessed(image_file.clone()))
                    .await;
            }
            let rating = rating_model.lock().unwrap().rate(&img)?;
            let result = pipe.lock().unwrap().predict(img, None)?;
            let simple_result = TaggingResultSimple::from(result);
            let hash = get_hash(&image_file)?;
            let size = fs::metadata(&image_file)?.len();
            if let Some(path_str) = image_file.to_str() {
                db.lock().unwrap().save_image_tags(
                    path_str,
                    size,
                    &hash,
                    &simple_result.tags,
                    rating.as_str(),
                )?;
            }
            tx.send(ProgressUpdate::Progress(
                0.25 + 0.375 * (i + 1) as f64 / total_images as f64,
            ))
            .await?;
        }
    }
    Ok(())
}

/// Processes all video files in the selected directories.
async fn process_videos(
    selected_dirs: &[PathBuf],
    pipe: &Arc<Mutex<TaggingPipeline>>,
    rating_model: &Arc<Mutex<RatingModel>>,
    db: &Arc<Mutex<Database>>,
    tx: &mpsc::Sender<ProgressUpdate>,
    show_ascii_art: bool,
) -> Result<()> {
    let mut video_files = Vec::new();
    for dir in selected_dirs {
        if let Some(dir_str) = dir.to_str() {
            video_files.extend(video::get_video_files(dir_str).await?);
        }
    }

    let total_videos = video_files.len();
    if total_videos > 0 {
        tx.send(ProgressUpdate::Message(format!(
            "Processing {} video files...",
            total_videos
        )))
        .await?;
        for (i, video_file) in video_files.into_iter().enumerate() {
            video::process_video(
                &video_file,
                pipe,
                rating_model,
                db,
                get_hash,
                tx,
                show_ascii_art,
            )
            .await?;
            tx.send(ProgressUpdate::Progress(
                0.625 + 0.375 * (i + 1) as f64 / total_videos as f64,
            ))
            .await?;
        }
    }
    Ok(())
}

/// Computes the SHA256 hash of a file.
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

/// Holds the configuration settings for the application.
#[derive(Debug, Default, Clone)]
pub struct AppConfig {
    pub model: V3Model,
    pub input_path: String,
    pub video_path: String,
    pub threshold: f32,
    pub batch_size: usize,
    pub show_ascii_art: bool,
}