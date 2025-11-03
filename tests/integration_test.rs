use eros::prelude::{
    convert_and_strip_metadata, rename_files_in_selected_dirs,
    suggest_media_directories,
};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

mod common;
use common::setup;

#[test]
fn test_full_preprocessing_pipeline() {
    // 1. Setup: Ensure assets are downloaded and create a temporary directory for the test.
    setup();
    let temp_dir = tempdir().unwrap();
    let assets_dir = Path::new("tests/assets");

    // Copy test files into the temporary directory
    let test_image_src = assets_dir.join("test_image.jpg");
    let temp_image_dst = temp_dir.path().join("test_image.jpg");

    fs::copy(test_image_src, &temp_image_dst).unwrap();

    let selected_dirs = vec![temp_dir.path().to_path_buf()];

    // 2. Test suggest_media_directories (on the temp dir)
    let suggested = suggest_media_directories(temp_dir.path()).unwrap();
    assert_eq!(suggested.len(), 1);
    assert_eq!(suggested[0], temp_dir.path());

    // 3. Test rename_files_in_selected_dirs
    rename_files_in_selected_dirs(&selected_dirs).unwrap();

    // Check that files are renamed. Assuming test_image.jpg comes before test_video.mp4 alphabetically
    let renamed_image_path = temp_dir.path().join("1.jpg");
    assert!(renamed_image_path.exists());

    // 4. Test convert_and_strip_metadata
    convert_and_strip_metadata(&selected_dirs).unwrap();

    let converted_image_path = temp_dir.path().join("1.png");
    assert!(converted_image_path.exists());
    assert!(!renamed_image_path.exists()); // Original should be deleted

}