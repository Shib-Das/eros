use crate::db::Database;
use crate::file::TaggingResultSimple;
use anyhow::Result;
use eros::pipeline::TaggingPipeline;
use futures::stream::{self, StreamExt};
use image::DynamicImage;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

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
    db: &Arc<Mutex<Database>>,
    get_hash_fn: impl Fn(&Path) -> Result<String>,
) -> Result<()> {
    println!("Processing video: {}", video_path.display());

    // Create a temporary folder for frames
    let video_name = video_path.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    let temp_frame_dir = format!("./temp_frames_{}", video_name);
    fs::create_dir_all(&temp_frame_dir)?;

    println!("  - Extracting frames to {}...", temp_frame_dir);

    // Extract frames every 3 seconds
    let extracted_frame_paths = extract_frames(video_path, &temp_frame_dir)?;
    println!("  - Extracted {} frames.", extracted_frame_paths.len());

    if extracted_frame_paths.is_empty() {
        println!("  - No frames extracted, skipping video.");
        fs::remove_dir_all(&temp_frame_dir)?;
        return Ok(());
    }

    // Run the AI tagger on all the frames
    println!("  - Tagging extracted frames...");
    let mut all_tags = Vec::new();
    let frame_images: Vec<DynamicImage> = extracted_frame_paths
        .iter()
        .filter_map(|p| image::open(p).ok())
        .collect();

    if !frame_images.is_empty() {
        let results = pipe
            .lock()
            .unwrap()
            .predict_batch(frame_images.iter().collect())?;
        for result in results {
            let simple_result = TaggingResultSimple::from(result);
            if !simple_result.tags.is_empty() {
                all_tags.extend(simple_result.tags.split(", ").map(|s| s.to_string()));
            }
        }
    }

    // Save the concatenated tags to the database
    let tags_string = all_tags.join(", ");
    let hash = get_hash_fn(video_path)?;
    let size = fs::metadata(video_path)?.len();

    {
        let db_lock = db.lock().unwrap();
        db_lock.save_video_tags(video_path.to_str().unwrap(), size, &hash, &tags_string)?;
        println!("  - Saved {} tags to the database.", all_tags.len());

        // Clean up the database by removing duplicate tags
        db_lock.cleanup_video_tags(&hash)?;
        println!("  - Cleaned up and deduplicated tags in the database.");
    }

    // Clean up temporary frames
    fs::remove_dir_all(&temp_frame_dir)?;
    println!("  - Removed temporary frame directory.");

    Ok(())
}

/// Extracts frames from a video at a 3-second interval.
fn extract_frames(video_path: &Path, output_dir: &str) -> Result<Vec<PathBuf>> {
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
    let mut saved_frame_count = 0;
    let mut extracted_frame_paths = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if frame_count % frame_interval == 0 {
                    let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    let frame_path =
                        PathBuf::from(output_dir).join(format!("frame_{:05}.png", saved_frame_count));

                    // --- FIX IS HERE ---
                    let width = rgb_frame.width() as usize;
                    let height = rgb_frame.height() as usize;
                    let stride = rgb_frame.stride(0) as usize;
                    let data = rgb_frame.data(0);

                    let mut image_data = Vec::with_capacity(width * height * 3);
                    if stride == width * 3 {
                        // No padding, can copy directly
                        image_data.extend_from_slice(data);
                    } else {
                        // Copy row by row to remove padding
                        for y in 0..height {
                            let start = y * stride;
                            let end = start + width * 3;
                            image_data.extend_from_slice(&data[start..end]);
                        }
                    }

                    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                        width as u32,
                        height as u32,
                        image_data,
                    )
                    .expect("Failed to create image buffer from frame data")
                    .save(&frame_path)?;
                    // --- END OF FIX ---

                    extracted_frame_paths.push(frame_path);
                    saved_frame_count += 1;
                }
                frame_count += 1;
            }
        }
    }
    Ok(extracted_frame_paths)
}