use anyhow::Result;
use eros::optimizer;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

mod common;

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