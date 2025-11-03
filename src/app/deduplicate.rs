use anyhow::Result;
 use image::{imageops::FilterType, DynamicImage};
 use std::{
 collections::{HashMap, HashSet},
 fs,
 path::PathBuf,
 };
 use tokio::sync::mpsc;
 use walkdir::WalkDir;

use crate::app::ProgressUpdate;
 use crate::file::is_image;

/// Calculates a 64-bit "fingerprint" of an image by resizing it to 8x8,
 /// converting it to grayscale, and creating a hash based on whether each pixel
 /// is brighter than the average.
 fn calculate_fingerprint(image: &DynamicImage) -> u64 {
    let resized = image.resize_exact(8, 8, FilterType::Triangle);
    let luma = resized.to_luma8();

    let pixels: Vec<u8> = luma.pixels().map(|p| p[0]).collect();
    let sum: u32 = pixels.iter().map(|&p| p as u32).sum();

    // Handle solid color edge case
    if pixels.iter().all(|&p| p == pixels[0]) {
        return if pixels[0] < 128 { 0 } else { u64::MAX };
    }

    let avg = (sum / 64) as u8;

    let mut hash = 0u64;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel >= avg {
            hash |= 1 << i;
        }
    }
    hash
}

/// Calculates the Hamming distance (number of differing bits) between two fingerprints.
 fn hamming_distance(hash1: u64, hash2: u64) -> u32 {
 (hash1 ^ hash2).count_ones()
 }

/// Removes duplicate images from the selected directories.
 pub async fn remove_duplicate_images(
    selected_dirs: &[PathBuf],
    tx: &mpsc::Sender<ProgressUpdate>,
) -> Result<()> {
    let mut image_files: Vec<PathBuf> = selected_dirs
        .iter()
        .flat_map(|dir| {
 WalkDir::new(dir)
 .into_iter()
 .filter_map(|e| e.ok())
 .filter(|e| {
 e.file_type().is_file()
 && is_image(e.path().to_str().unwrap_or("")).unwrap_or(false) //
 })
 .map(|e| e.path().to_path_buf())
 })
 .collect();

    image_files.sort();

if image_files.len() < 2 {
 return Ok(());
 }

let mut fingerprints = HashMap::new();
 for path in &image_files {
 if let Ok(image) = image::open(path) {
 fingerprints.insert(path.clone(), calculate_fingerprint(&image));
 }
 }

let mut duplicates_to_remove = HashSet::new();
 for i in 0..image_files.len() {
 for j in (i + 1)..image_files.len() {
 let path1 = &image_files[i];
 let path2 = &image_files[j];

if duplicates_to_remove.contains(path1) || duplicates_to_remove.contains(path2) {
 continue;
 }

if let (Some(&hash1), Some(&hash2)) = (fingerprints.get(path1), fingerprints.get(path2)) {
 let distance = hamming_distance(hash1, hash2);
 // 90% similarity is a Hamming distance of <= 6 for a 64-bit hash.
 if distance <= 6 {
 duplicates_to_remove.insert(path2.clone());
 let similarity = 100.0 * (1.0 - (distance as f64 / 64.0));
 let message = format!(
 "Duplicate found: {:?} is {:.1}% similar to {:?}. Removing.",
 path2.file_name().unwrap(),
 similarity,
 path1.file_name().unwrap()
 );
 tx.send(ProgressUpdate::Message(message)).await?;
 }
 }
 }
 }

for file_path in duplicates_to_remove {
 if fs::remove_file(&file_path).is_ok() {
 let message = format!("Removed duplicate: {:?}", file_path);
 tx.send(ProgressUpdate::Message(message)).await?;
 }
 }

Ok(())
 }

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0b1111, 0b0000), 4);
        assert_eq!(hamming_distance(0b1010, 0b1010), 0);
        assert_eq!(hamming_distance(0b11111111, 0b00000000), 8);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    #[test]
    fn test_fingerprint_consistency() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([128, 128, 128])));
        let fingerprint1 = calculate_fingerprint(&img);
        let fingerprint2 = calculate_fingerprint(&img);
        assert_eq!(fingerprint1, fingerprint2);
    }

    #[test]
    fn test_fingerprint_difference() {
        let img1 = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([0, 0, 0]))); // Black
        let img2 = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([255, 255, 255]))); // White
        let fingerprint1 = calculate_fingerprint(&img1);
        let fingerprint2 = calculate_fingerprint(&img2);
        // A black and white image should have a large hamming distance
        assert!(hamming_distance(fingerprint1, fingerprint2) > 32);
    }
}
