use crate::{
    app::ProgressUpdate,
    db::Database,
    file::{TaggingResultSimple, TaggingResultSimpleTags},
};
use anyhow::{Context, Result};
use eros::{pipeline::TaggingPipeline, rating::RatingModel};
use futures::stream::{self, StreamExt};
use image::{DynamicImage, GrayImage};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

/// Supported video extensions.
pub const VIDEO_EXTENSIONS: [&str; 4] = ["mp4", "mkv", "webm", "avi"];

/// Check if the path is a video file.
pub fn is_video(path: &str) -> Result<bool> {
    match PathBuf::from(path).extension() {
        Some(ext) => {
            let ext = ext.to_string_lossy().to_lowercase();
            Ok(VIDEO_EXTENSIONS.contains(&ext.as_str()))
        }
        None => Ok(false),
    }
}

/// Get video files from a directory.
pub async fn get_video_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut entries = tokio::fs::read_dir(dir).await?;
    let mut tasks = vec![];

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let task = tokio::spawn(async move {
            if is_video(path.to_str()?).unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        });

        tasks.push(task);
    }

    let files = stream::iter(tasks)
        .buffer_unordered(16)
        .filter_map(|result| async move { result.ok().flatten() })
        .collect()
        .await;

    Ok(files)
}

/// Processes a single video file by extracting frames, tagging them, and storing the results.
pub async fn process_video(
    video_path: &Path,
    pipe: &Arc<Mutex<TaggingPipeline>>,
    rating_model: &Arc<Mutex<RatingModel>>,
    get_hash_fn: impl Fn(&Path) -> Result<String>,
    tx: &mpsc::Sender<ProgressUpdate>,
    show_ascii_art: bool,
) -> Result<TaggingResultSimple> {
    // Extract frames every 3 seconds
    let frame_images = extract_frames(video_path)?;

    if frame_images.is_empty() {
        anyhow::bail!("No frames extracted from video");
    }

    let mut all_tags = Vec::new();
    let mut overall_rating = "sfw";

    for frame_image in frame_images {
        if show_ascii_art {
            if tx.send(ProgressUpdate::Frame(frame_image.clone())).await.is_err() {
                // UI receiver has been dropped, so we can stop.
                anyhow::bail!("UI closed");
            }
        }

        // Determine rating, stopping at the first NSFW frame
        if overall_rating != "nsfw" {
            let rating = rating_model.lock().unwrap().rate(&frame_image)?;
            if rating.as_str() == "nsfw" {
                overall_rating = "nsfw";
            }
        }

        let result = pipe.lock().unwrap().predict(frame_image, None)?;
        let character_tags = result
            .character
            .keys()
            .map(|tag| super::tag::fix_tag_underscore(tag));
        all_tags.extend(character_tags);

        let general_tags = result
            .general
            .keys()
            .map(|tag| super::tag::fix_tag_underscore(tag));
        all_tags.extend(general_tags);
    }

    // Save the concatenated tags to the database
    let tags_string = all_tags.join(", ");
    let hash = get_hash_fn(video_path)?;
    let size = fs::metadata(video_path)?.len();

    let tagger_result = TaggingResultSimple {
        filename: video_path.to_str().unwrap().to_string(),
        size,
        hash,
        tags: tags_string,
        rating: overall_rating.to_string(),
        tagger: TaggingResultSimpleTags {
            rating: overall_rating.to_string(),
            character: Vec::new(),
            general: all_tags,
        },
    };

    Ok(tagger_result)
}

/// Extracts frames from a video based on scene changes.
pub fn extract_frames(video_path: &Path) -> Result<Vec<DynamicImage>> {
    ffmpeg_next::init().unwrap();
    let mut ictx = ffmpeg_next::format::input(&video_path)?;
    let input = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or(ffmpeg_next::Error::StreamNotFound)?;
    let video_stream_index = input.index();

    let context_decoder =
        ffmpeg_next::codec::context::Context::from_parameters(input.parameters())?;
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

    let mut extracted_frames = Vec::new();
    let mut last_grayscale_frame: Option<GrayImage> = None;

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;

                let width = rgb_frame.width();
                let height = rgb_frame.height();
                let data = rgb_frame.data(0).to_vec();

                let image_buffer =
                    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, data)
                        .context("Failed to create image buffer")?;
                let dynamic_image = DynamicImage::ImageRgb8(image_buffer);
                let grayscale_frame = dynamic_image.to_luma8();

                let should_extract = if let Some(last_frame) = &last_grayscale_frame {
                    const THRESHOLD: f64 = 0.1;
                    let diff = frame_difference(last_frame, &grayscale_frame);
                    diff > THRESHOLD
                } else {
                    true
                };

                if should_extract {
                    extracted_frames.push(dynamic_image);
                    last_grayscale_frame = Some(grayscale_frame);
                }
            }
        }
    }
    Ok(extracted_frames)
}

/// Calculates the mean absolute difference between two grayscale frames.
fn frame_difference(frame1: &GrayImage, frame2: &GrayImage) -> f64 {
    let diff: f64 = frame1
        .pixels()
        .zip(frame2.pixels())
        .map(|(p1, p2)| (p1[0] as f64 - p2[0] as f64).abs())
        .sum();
    diff / (frame1.width() * frame1.height()) as f64
}