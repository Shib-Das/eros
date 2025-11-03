//! This module provides functions for optimizing media files.
//!
//! It includes utilities for compressing images (PNG, JPEG) and videos (MP4)
//! to reduce their file size while maintaining quality. The optimizations are
//! designed to be applied after all other processing is complete.

use anyhow::{Context, Result};
use mozjpeg::{ColorSpace, Compress, Decompress};
use oxipng::{optimize, InFile, Options, OutFile};
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;
use walkdir::WalkDir;

/// Optimizes a single image file.
///
/// This function will re-compress JPEGs and PNGs to reduce their file size.
/// It saves the optimized file to a temporary location and then replaces the original
/// to ensure the operation is atomic.
fn optimize_image(path: &Path) -> Result<()> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_lowercase();

    match extension.as_str() {
        "jpg" | "jpeg" => optimize_jpeg(path),
        "png" => optimize_png(path),
        _ => Ok(()),
    }
}

/// Optimizes a JPEG file by re-compressing it.
fn optimize_jpeg(path: &Path) -> Result<()> {
    let file_data =
        fs::read(path).with_context(|| format!("Failed to read image file: {:?}", path))?;

    let mut image = Decompress::new_mem(&file_data)?
        .rgb()
        .with_context(|| "Failed to decompress to RGB")?;
    let (width, height) = (image.width(), image.height());

    let mut compress = Compress::new(ColorSpace::JCS_RGB);
    compress.set_quality(75.0);
    compress.set_size(width, height);

    let mut comp = compress
        .start_compress(Vec::new())
        .with_context(|| "Failed to start compression")?;
    comp.write_scanlines(rgb::ComponentBytes::as_bytes(
        image.read_scanlines::<rgb::RGB8>()?.as_slice(),
    ))
    .with_context(|| "Failed to write scanlines")?;
    let compressed_data = comp.finish()?;

    let temp_file = NamedTempFile::new_in(
        path.parent()
            .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))?,
    )?;
    fs::write(temp_file.path(), &compressed_data)?;
    temp_file
        .persist(path)
        .map_err(|e| e.error)
        .with_context(|| format!("Failed to replace original file: {:?}", path))?;

    Ok(())
}

/// Optimizes a PNG file using `oxipng`.
fn optimize_png(path: &Path) -> Result<()> {
    let options = Options::from_preset(2);
    let in_file = InFile::Path(path.to_path_buf());
    let temp_file = NamedTempFile::new_in(
        path.parent()
            .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))?,
    )?;
    let out_file = OutFile::Path {
        path: Some(temp_file.path().to_path_buf()),
        preserve_attrs: true,
    };

    optimize(&in_file, &out_file, &options)
        .with_context(|| format!("Failed to optimize PNG: {:?}", path))?;

    temp_file
        .persist(path)
        .map_err(|e| e.error)
        .with_context(|| format!("Failed to replace original file: {:?}", path))?;

    Ok(())
}


/// Optimizes all media files in the given directories.
pub async fn optimize_media_in_dirs(dirs: &[PathBuf]) -> Result<()> {
    let media_files: Vec<PathBuf> = dirs
        .par_iter()
        .flat_map(|dir| {
            WalkDir::new(dir)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.path().is_file())
                .map(|e| e.path().to_path_buf())
                .collect::<Vec<PathBuf>>()
        })
        .collect();

    media_files.par_iter().try_for_each(|path| {
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        match extension.as_str() {
            "jpg" | "jpeg" | "png" => {
                optimize_image(path).with_context(|| format!("Failed to optimize image: {:?}", path))
            }
            _ => Ok(()),
        }
    })
}