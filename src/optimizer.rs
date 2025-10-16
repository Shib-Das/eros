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

use ffmpeg_next as ffmpeg;

/// Encodes a single frame to the output context.
fn encode_frame(
    encoder: &mut ffmpeg::encoder::video::Video,
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
    encoder: &mut ffmpeg::encoder::video::Video,
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

    let best_video_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .map(|s| s.index());

    let (stream_mapping, mut video_encoder, mut sws_context) =
        setup_streams(&mut ictx, &mut octx, best_video_stream_index)?;

    octx.write_header()?;

    for (stream, packet) in ictx.packets() {
        let istream_index = stream.index();
        let ostream_index = stream_mapping[istream_index];

        if Some(istream_index) == best_video_stream_index {
            if let (Some((ref mut enc, ref mut dec)), Some(ref mut scaler)) =
                (video_encoder.as_mut(), sws_context.as_mut())
            {
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

    if let (Some((ref mut enc, ref mut dec)), Some(ref mut scaler)) =
        (video_encoder.as_mut(), sws_context.as_mut())
    {
        dec.send_eof()?;
        let mut decoded = ffmpeg::frame::Video::empty();
        let time_base = dec.time_base();
        while dec.receive_frame(&mut decoded).is_ok() {
            let mut scaled = ffmpeg::frame::Video::empty();
            scaler.run(&decoded, &mut scaled)?;
            scaled.set_pts(decoded.pts());
            encode_frame(
                enc,
                &scaled,
                &mut octx,
                stream_mapping[best_video_stream_index.unwrap()],
                time_base,
            )?;
        }
        flush_encoder(
            enc,
            &mut octx,
            stream_mapping[best_video_stream_index.unwrap()],
        )?;
    }

    octx.write_trailer()?;

    temp_file
        .persist(path)
        .map_err(|e| e.error)
        .with_context(|| format!("Failed to replace original file at {:?}", path))?;

    Ok(())
}

#[allow(clippy::type_complexity)]
fn setup_streams(
    ictx: &mut ffmpeg::format::context::Input,
    octx: &mut ffmpeg::format::context::Output,
    best_video_stream_index: Option<usize>,
) -> Result<(
    Vec<usize>,
    Option<(
        ffmpeg::encoder::Video,
        ffmpeg::decoder::video::Video,
    )>,
    Option<ffmpeg::software::scaling::Context>,
)> {
    let mut stream_mapping = vec![0; ictx.nb_streams() as usize];
    let mut video_encoder = None;
    let mut sws_context = None;

    let format_requires_global_header = octx
        .format()
        .flags()
        .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);

    for (istream_index, istream) in ictx.streams().enumerate() {
        let istream_params = istream.parameters();
        let mut ostream = octx.add_stream(None)?;
        ostream.set_parameters(istream_params.clone());

        if Some(istream_index) == best_video_stream_index {
            let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::H264)
                .context("Failed to find H.264 encoder")?;
            let dec = ffmpeg::codec::context::Context::from_parameters(istream.parameters())?
                .decoder()
                .video()?;

            let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
                .encoder()
                .video()?;
            encoder.set_height(dec.height());
            encoder.set_width(dec.width());
            encoder.set_format(ffmpeg::format::Pixel::YUV420P);
            let mut time_base = istream.time_base();
            if time_base.1 > 65535 {
                time_base = ffmpeg::Rational::new(1, 30000);
            }
            encoder.set_time_base(time_base);
            if format_requires_global_header {
                encoder.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
            }

            if istream.avg_frame_rate() > ffmpeg::Rational::new(0, 1) {
                encoder.set_frame_rate(Some(istream.avg_frame_rate()));
            }

            let mut opts = ffmpeg::Dictionary::new();
            opts.set("preset", "medium");
            opts.set("crf", "23");

            let enc = encoder.open_with(opts)?;
            ostream.set_parameters(&enc);

            let scaler = ffmpeg::software::scaling::Context::get(
                dec.format(),
                dec.width(),
                dec.height(),
                enc.format(),
                enc.width(),
                enc.height(),
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )?;
            video_encoder = Some((enc, dec));
            sws_context = Some(scaler);
        }
        stream_mapping[istream_index] = ostream.index();
    }

    Ok((stream_mapping, video_encoder, sws_context))
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
            "mp4" | "mov" | "avi" | "mkv" | "webm" => {
                match optimize_video(path) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        eprintln!("Warning: Failed to optimize video {:?}: {}", path, e);
                        Ok(()) // Continue processing other files
                    }
                }
            }
            _ => Ok(()),
        }
    })
}