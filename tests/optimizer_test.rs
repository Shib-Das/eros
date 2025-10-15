use anyhow::Result;
use eros::optimizer;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

mod common;

#[tokio::test]
async fn test_optimize_video_reduces_size() -> Result<()> {
    common::setup();

    let temp_dir = tempdir()?;
    let video_path = PathBuf::from("tests/assets/test_video.mp4");
    let test_video_path = temp_dir.path().join("test_video.mp4");
    fs::copy(&video_path, &test_video_path)?;

    let original_size = fs::metadata(&test_video_path)?.len();

    let dirs = vec![temp_dir.path().to_path_buf()];
    optimizer::optimize_media_in_dirs(&dirs).await?;

    let optimized_size = fs::metadata(&test_video_path)?.len();

    assert!(
        optimized_size < original_size,
        "Optimized video should be smaller. Original: {}, Optimized: {}",
        original_size,
        optimized_size
    );
    assert!(optimized_size > 0, "Optimized video should not be empty");

    Ok(())
}

use eros::prelude::resize_media;

#[tokio::test]
async fn test_optimize_image_reduces_size() -> Result<()> {
    common::setup();

    let temp_dir = tempdir()?;
    let image_path = PathBuf::from("tests/assets/test_image.jpg");
    let test_image_path = temp_dir.path().join("test_image.jpg");
    fs::copy(&image_path, &test_image_path)?;

    let original_size = fs::metadata(&test_image_path)?.len();

    let dirs = vec![temp_dir.path().to_path_buf()];
    optimizer::optimize_media_in_dirs(&dirs).await?;

    let optimized_size = fs::metadata(&test_image_path)?.len();

    assert!(
        optimized_size < original_size,
        "Optimized image should be smaller. Original: {}, Optimized: {}",
        original_size,
        optimized_size
    );
    assert!(optimized_size > 0, "Optimized image should not be empty");

    Ok(())
}

#[tokio::test]
async fn test_resize_video_dimensions() -> Result<()> {
    common::setup();
    ffmpeg_next::init().unwrap();

    let temp_dir = tempdir()?;
    let video_path = PathBuf::from("tests/assets/test_video.mp4");
    let test_video_path = temp_dir.path().join("test_video.mp4");
    fs::copy(&video_path, &test_video_path)?;

    let dirs = vec![temp_dir.path().to_path_buf()];
    let target_size = (256, 256);
    resize_media(&dirs, target_size)?;

    let ictx = ffmpeg_next::format::input(&test_video_path)?;
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let decoder_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())?;
    let video = decoder_ctx.decoder().video()?;

    assert_eq!(video.width(), target_size.0);
    assert_eq!(video.height(), target_size.1);

    Ok(())
}