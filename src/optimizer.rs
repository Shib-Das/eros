//! This module provides functions for optimizing media files.
//!
//! It includes utilities for compressing images (PNG, JPEG) and videos (MP4)
//! to reduce their file size while maintaining quality. The optimizations are
//! designed to be applied after all other processing is complete.

use anyhow::{Context, Result};
use mozjpeg::{ColorSpace, Compress, Decompress};
use oxipng::{optimize, InFile, Options, OutFile};
use rayon::prelude::*;
use rgb::ComponentBytes;
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
        "jpg" | "jpeg" => {
            let file_data =
                fs::read(path).with_context(|| format!("Failed to read image file: {:?}", path))?;

            // Decompress the image to raw pixels
            let decompress = Decompress::new_mem(&file_data)
                .with_context(|| "Failed to create JPEG decompressor")?;
            let mut image = decompress
                .rgb()
                .with_context(|| "Failed to decompress to RGB")?;
            let (width, height) = (image.width(), image.height());
            let pixels: Vec<rgb::RGB8> = image
                .read_scanlines()
                .with_context(|| "Failed to read scanlines")?;
            image
                .finish()
                .with_context(|| "Failed to finish decompression")?;

            // Re-compress the image with optimized settings
            let mut compress = Compress::new(ColorSpace::JCS_RGB);
            compress.set_quality(75.0);
            compress.set_size(width, height);
            let mut comp = compress
                .start_compress(Vec::new())
                .with_context(|| "Failed to start compression")?;
            comp.write_scanlines(pixels.as_slice().as_bytes())
                .with_context(|| "Failed to write scanlines")?;
            let compressed_data =
                comp.finish().with_context(|| "Failed to finish compression")?;

            // Write to a temporary file and then atomically replace the original
            let temp_file = NamedTempFile::new_in(
                path.parent()
                    .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))?,
            )
            .with_context(|| "Failed to create temporary file")?;
            fs::write(temp_file.path(), &compressed_data)
                .with_context(|| "Failed to write to temporary file")?;
            temp_file
                .persist(path)
                .map_err(|e| e.error)
                .with_context(|| format!("Failed to replace original file: {:?}", path))?;
            Ok(())
        }
        "png" => {
            let options = Options::from_preset(2);
            let in_file = InFile::Path(path.to_path_buf());
            let temp_file = NamedTempFile::new_in(
                path.parent()
                    .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))?,
            )
            .with_context(|| "Failed to create temporary file")?;

            // The `OutFile::Path` variant is a struct.
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
        _ => Ok(()), // Not a supported image format, so we do nothing.
    }
}

/// Finds all image files in the given directories and optimizes them in parallel.
pub fn optimize_images_in_dirs(dirs: &[PathBuf]) -> Result<()> {
    let mut image_files = Vec::new();
    for dir in dirs {
        for entry in WalkDir::new(dir) {
            let entry = entry.with_context(|| "Failed to read directory entry")?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                        image_files.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    image_files.par_iter().try_for_each(|path| {
        optimize_image(path).with_context(|| format!("Failed to optimize image: {:?}", path))
    })
}

use ffmpeg_next as ffmpeg;

/// Encodes a single frame to the output context.
fn encode_frame(
    encoder: &mut ffmpeg::encoder::Video,
    frame: &ffmpeg::frame::Video,
    octx: &mut ffmpeg::format::context::Output,
    ostream_index: usize,
    input_time_base: ffmpeg::Rational,
) -> Result<()> {
    encoder.send_frame(frame)?;
    let mut encoded = ffmpeg::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(ostream_index);
        encoded.rescale_ts(input_time_base, encoder.time_base());
        encoded.write_interleaved(octx)?;
    }
    Ok(())
}

/// Flushes the encoder to ensure all frames are written.
fn flush_encoder(
    encoder: &mut ffmpeg::encoder::Video,
    octx: &mut ffmpeg::format::context::Output,
    ostream_index: usize,
) -> Result<()> {
    encoder.send_eof()?;
    let mut encoded = ffmpeg::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(ostream_index);
        encoded.write_interleaved(octx)?;
    }
    Ok(())
}


/// Optimizes a single video file by re-encoding it with H.264 and AAC.
fn optimize_video(path: &Path) -> Result<()> {
    let temp_file = tempfile::Builder::new()
        .suffix(".mp4")
        .tempfile_in(path.parent().ok_or_else(|| anyhow::anyhow!("Invalid path"))?)
        .with_context(|| "Failed to create temporary file")?;

    let mut ictx = ffmpeg::format::input(path)?;
    let mut octx = ffmpeg::format::output(&temp_file.path())?;
    octx.set_metadata(ictx.metadata().iter().collect());

    let best_video_stream = ictx.streams().best(ffmpeg::media::Type::Video).map(|s| s.index());

    let mut stream_mapping = vec![0; ictx.nb_streams() as usize];
    let mut video_encoder = None;
    let mut sws_context = None;

    let format_requires_global_header = octx
        .format()
        .flags()
        .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);

    for (istream_index, istream) in ictx.streams().enumerate() {
        if Some(istream_index) == best_video_stream {
            let mut ostream = octx.add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::MPEG4))?;
            let mut enc = ffmpeg::codec::context::Context::from_parameters(ostream.parameters())?
                .encoder()
                .video()?;
            let dec = ffmpeg::codec::context::Context::from_parameters(istream.parameters())?
                .decoder()
                .video()?;

            enc.set_height(dec.height());
            enc.set_width(dec.width());
            enc.set_format(ffmpeg::format::Pixel::YUV420P);
            enc.set_time_base(istream.time_base());
            if istream.avg_frame_rate() > ffmpeg::Rational::new(0, 1) {
                enc.set_frame_rate(Some(istream.avg_frame_rate()));
            }

            if format_requires_global_header {
                enc.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
            }

            let opened_encoder = enc.open_as(ffmpeg::codec::Id::MPEG4)?;
            ostream.set_parameters(&opened_encoder);

            let scaler = ffmpeg::software::scaling::Context::get(
                dec.format(), dec.width(), dec.height(),
                opened_encoder.format(), opened_encoder.width(), opened_encoder.height(),
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )?;

            stream_mapping[istream_index] = ostream.index();
            video_encoder = Some((opened_encoder, dec));
            sws_context = Some(scaler);
        } else {
            let mut ostream = octx.add_stream(None)?;
            ostream.set_parameters(istream.parameters());
            stream_mapping[istream_index] = ostream.index();
        }
    }

    octx.write_header()?;

    for (stream, packet) in ictx.packets() {
        let istream_index = stream.index();
        let ostream_index = stream_mapping[istream_index];

        if Some(istream_index) == best_video_stream {
            if let (Some((ref mut enc, ref mut dec)), Some(ref mut scaler)) = (video_encoder.as_mut(), sws_context.as_mut()) {
                dec.send_packet(&packet)?;
                let mut decoded = ffmpeg::frame::Video::empty();
                let time_base = dec.time_base();
                while dec.receive_frame(&mut decoded).is_ok() {
                    let mut scaled = ffmpeg::frame::Video::empty();
                    scaler.run(&decoded, &mut scaled)?;
                    scaled.set_pts(decoded.pts());
                    encode_frame(enc, &scaled, &mut octx, ostream_index, time_base)?;
                }
            }
        } else {
            let mut p = packet.clone();
            p.set_stream(ostream_index);
            p.write_interleaved(&mut octx)?;
        }
    }

    if let (Some((ref mut enc, ref mut dec)), Some(ref mut scaler)) = (video_encoder.as_mut(), sws_context.as_mut()) {
        dec.send_eof()?;
        let mut decoded = ffmpeg::frame::Video::empty();
        let time_base = dec.time_base();
        while dec.receive_frame(&mut decoded).is_ok() {
            let mut scaled = ffmpeg::frame::Video::empty();
            scaler.run(&decoded, &mut scaled)?;
            scaled.set_pts(decoded.pts());
            encode_frame(enc, &scaled, &mut octx, stream_mapping[best_video_stream.unwrap()], time_base)?;
        }
        flush_encoder(enc, &mut octx, stream_mapping[best_video_stream.unwrap()])?;
    }

    octx.write_trailer()?;

    temp_file
        .persist(path)
        .map_err(|e| e.error)
        .with_context(|| format!("Failed to replace original file at {:?}", path))?;

    Ok(())
}


/// Finds all video files and optimizes them in parallel.
pub fn optimize_videos_in_dirs(dirs: &[PathBuf]) -> Result<()> {
    let mut video_files = Vec::new();
    for dir in dirs {
        for entry in WalkDir::new(dir) {
            let entry = entry.with_context(|| "Failed to read directory entry")?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if matches!(
                        ext.to_lowercase().as_str(),
                        "mp4" | "mov" | "avi" | "mkv" | "webm"
                    ) {
                        video_files.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    video_files.par_iter().try_for_each(|path| {
        optimize_video(path).with_context(|| format!("Failed to optimize video: {:?}", path))
    })
}


/// Optimizes all media files in the given directories.
pub async fn optimize_media_in_dirs(dirs: &[PathBuf]) -> Result<()> {
    optimize_images_in_dirs(dirs)?;
    optimize_videos_in_dirs(dirs)?;
    Ok(())
}