use crate::{
    app::ProgressUpdate,
    db::Database,
    file::{TaggingResultSimple, TaggingResultSimpleTags},
};
use anyhow::Result;
use eros::{pipeline::TaggingPipeline, rating::RatingModel};
use futures::stream::{self, StreamExt};
use image::DynamicImage;
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

/// Extracts frames from a video at a 3-second interval.
fn extract_frames(video_path: &Path) -> Result<Vec<DynamicImage>> {
    ffmpeg_next::init().unwrap();
    let mut ictx = ffmpeg_next::format::input(&video_path)?;
    let input = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or(ffmpeg_next::Error::StreamNotFound)?;
    let video_stream_index = input.index();
    let frame_rate = input.avg_frame_rate();
    let frame_interval = (frame_rate.0 as f64 / frame_rate.1 as f64 * 3.0).round() as i64;

    if frame_interval == 0 {
        return Err(anyhow::anyhow!("Invalid frame interval for video."));
    }

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

    let mut frame_count = 0i64;
    let mut extracted_frames = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if frame_count % frame_interval == 0 {
                    let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;

                    let width = rgb_frame.width() as usize;
                    let height = rgb_frame.height() as usize;
                    let stride = rgb_frame.stride(0) as usize;
                    let data = rgb_frame.data(0);

                    let mut image_data = Vec::with_capacity(width * height * 3);
                    if stride == width * 3 {
                        image_data.extend_from_slice(data);
                    } else {
                        for y in 0..height {
                            let start = y * stride;
                            let end = start + width * 3;
                            image_data.extend_from_slice(&data[start..end]);
                        }
                    }

                    if let Some(image_buffer) = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                        width as u32,
                        height as u32,
                        image_data,
                    ) {
                        extracted_frames.push(DynamicImage::ImageRgb8(image_buffer));
                    }
                }
                frame_count += 1;
            }
        }
    }
    Ok(extracted_frames)
}