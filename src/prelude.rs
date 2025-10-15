pub use crate::error::{ErosError, Result};
use ffmpeg_next as ffmpeg;
use std::{
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "webp"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "avi", "mkv", "mov", "webm"];

pub fn suggest_media_directories(start_path: &Path) -> Result<Vec<PathBuf>> {
    let mut media_dirs = Vec::new();

    for entry in WalkDir::new(start_path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                if IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str())
                    || VIDEO_EXTENSIONS.contains(&ext.to_lowercase().as_str())
                {
                    if let Some(parent) = entry.path().parent() {
                        if !media_dirs.contains(&parent.to_path_buf()) {
                            media_dirs.push(parent.to_path_buf());
                        }
                    }
                }
            }
        }
    }

    Ok(media_dirs)
}

pub fn rename_files_in_selected_dirs(selected_dirs: &[PathBuf]) -> Result<()> {
    let mut counter = 1;
    for dir in selected_dirs {
        let mut entries: Vec<_> = WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();

        // Sort entries to ensure deterministic renaming
        entries.sort_by_key(|e| e.path().to_path_buf());

        for entry in entries {
            if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                let new_name = format!("{}.{}", counter, ext);
                let new_path = entry.path().with_file_name(new_name);
                fs::rename(entry.path(), new_path)?;
                counter += 1;
            }
        }
    }
    Ok(())
}

pub fn convert_and_strip_metadata(selected_dirs: &[PathBuf]) -> Result<()> {
    for dir in selected_dirs {
        let entries: Vec<_> = WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();

        for entry in entries {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let ext_lower = ext.to_lowercase();

                if IMAGE_EXTENSIONS.contains(&ext_lower.as_str()) {
                    let img = image::open(path)?;
                    let new_path = path.with_extension("png");
                    img.save(&new_path)?;
                    if path != new_path {
                        fs::remove_file(path)?;
                    }
                } else if VIDEO_EXTENSIONS.contains(&ext_lower.as_str()) {
                    let new_path = path.with_extension("mp4");
                    if path.as_os_str() != new_path.as_os_str() {
                        remux(path, &new_path)?;
                        fs::remove_file(path)?;
                    } else {
                        // It's already an MP4, but we need to strip metadata.
                        let temp_output_path = path.with_extension("temp.mp4");
                        remux(path, &temp_output_path)?;
                        fs::remove_file(path)?;
                        fs::rename(&temp_output_path, path)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn remux(from: &Path, to: &Path) -> Result<()> {
    let mut ictx = ffmpeg::format::input(&from).map_err(|e| ErosError::Optimizer(e.to_string()))?;
    let mut octx = ffmpeg::format::output_as(&to, "mp4").map_err(|e| ErosError::Optimizer(e.to_string()))?;

    octx.set_metadata(Default::default());

    let mut stream_mapping = vec![0; ictx.nb_streams() as usize];
    for in_stream in ictx.streams() {
        let mut out_stream = octx.add_stream(None).map_err(|e| ErosError::Optimizer(e.to_string()))?;
        out_stream.set_parameters(in_stream.parameters());
        out_stream.set_metadata(Default::default());
        stream_mapping[in_stream.index()] = out_stream.index();
    }

    octx.write_header().map_err(|e| ErosError::Optimizer(e.to_string()))?;

    for (stream, mut packet) in ictx.packets() {
        let out_stream_index = stream_mapping[stream.index()];
        let out_stream = octx.stream(out_stream_index).unwrap();

        packet.rescale_ts(stream.time_base(), out_stream.time_base());
        packet.set_stream(out_stream_index);
        packet.write_interleaved(&mut octx).map_err(|e| ErosError::Optimizer(e.to_string()))?;
    }

    octx.write_trailer().map_err(|e| ErosError::Optimizer(e.to_string()))?;
    Ok(())
}

pub fn resize_media(selected_dirs: &[PathBuf], size: (u32, u32)) -> Result<()> {
    for dir in selected_dirs {
        let entries: Vec<_> = WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();

        for entry in entries {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let ext_lower = ext.to_lowercase();

                if IMAGE_EXTENSIONS.contains(&ext_lower.as_str()) {
                    let img = image::open(path)?;
                    let resized_img = img.resize_exact(size.0, size.1, image::imageops::FilterType::Triangle);
                    resized_img.save(path)?;
                } else if VIDEO_EXTENSIONS.contains(&ext_lower.as_str()) {
                    let temp_path = path.with_extension("resized.mp4");
                    resize_video(path, &temp_path, size)?;
                    fs::remove_file(path)?;
                    fs::rename(&temp_path, path.with_extension("mp4"))?;
                }
            }
        }
    }
    Ok(())
}

fn resize_video(from: &Path, to: &Path, size: (u32, u32)) -> Result<()> {
    ffmpeg::init().unwrap();

    // Enable trace logging for libx264 debugging
    ffmpeg::log::set_level(ffmpeg::log::Level::Trace);

    let mut ictx = ffmpeg::format::input(&from)?;
    let mut octx = ffmpeg::format::output(&to)?;

    // Find input video stream
    let input_stream = ictx.streams().best(ffmpeg::media::Type::Video).ok_or(ffmpeg::Error::StreamNotFound)?;
    let input_time_base = input_stream.time_base();
    let video_index = input_stream.index();
    let fps = input_stream.avg_frame_rate();

    // Decoder setup using Context::from_parameters (no .codec() on stream)
    let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())?;
    let mut decoder = decoder_ctx.decoder().video()?;

    // Encoder codec: Use libx264 specifically
    let codec = ffmpeg::encoder::find_by_name("libx264").ok_or(ffmpeg::Error::EncoderNotFound)?;

    // Add output stream and get its index (short-lived borrow)
    let output_stream_index = octx.add_stream(codec)?.index();

    let output_time_base = fps.invert();

    // Short-lived mut borrow for setting time base
    octx.stream_mut(output_stream_index)
        .expect("Stream not found")
        .set_time_base(output_time_base);

    // Encoder setup
    let mut enc = ffmpeg::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()?;

    // Configure encoder
    enc.set_width(size.0);
    enc.set_height(size.1);
    enc.set_format(ffmpeg::format::Pixel::YUV420P);
    enc.set_frame_rate(Some(fps));
    enc.set_time_base(output_time_base);
    enc.set_bit_rate(2_000_000);

    // Dictionary options
    let mut opts = ffmpeg::Dictionary::new();
    opts.set("preset", "medium");
    opts.set("crf", "23");

    // Open encoder
    let mut opened_enc = enc.open_as_with(codec, opts)?;

    // Copy parameters from the opened encoder to the output stream
    octx.stream_mut(output_stream_index)
        .expect("Stream not found")
        .set_parameters(&opened_enc);

    // Global header if needed (check via immutable borrow)
    if octx.format().flags().contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER) {
        opened_enc.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
    }

    // Write header (mut borrow on octx)
    octx.write_header()?;

    // Scaler
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        opened_enc.format(),
        opened_enc.width(),
        opened_enc.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;

    // Processing loop (remove mut from packet as it's not mutated)
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_index {
            decoder.send_packet(&packet)?;

            let mut decoded_frame = ffmpeg::frame::Video::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                let mut resized_frame = ffmpeg::frame::Video::empty();
                scaler.run(&decoded_frame, &mut resized_frame)?;

                resized_frame.set_pts(decoded_frame.pts());

                opened_enc.send_frame(&resized_frame)?;

                let mut encoded_packet = ffmpeg::Packet::empty();
                while opened_enc.receive_packet(&mut encoded_packet).is_ok() {
                    encoded_packet.set_stream(output_stream_index);
                    encoded_packet.rescale_ts(input_time_base, output_time_base);
                    encoded_packet.write_interleaved(&mut octx)?;
                }
            }
        }
    }

    // Flush decoder
    decoder.send_eof()?;
    let mut decoded_frame = ffmpeg::frame::Video::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resized_frame = ffmpeg::frame::Video::empty();
        scaler.run(&decoded_frame, &mut resized_frame)?;
        resized_frame.set_pts(decoded_frame.pts());
        opened_enc.send_frame(&resized_frame)?;

        let mut encoded_packet = ffmpeg::Packet::empty();
        while opened_enc.receive_packet(&mut encoded_packet).is_ok() {
            encoded_packet.set_stream(output_stream_index);
            encoded_packet.rescale_ts(input_time_base, output_time_base);
            encoded_packet.write_interleaved(&mut octx)?;
        }
    }

    // Flush encoder
    opened_enc.send_eof()?;
    let mut encoded_packet = ffmpeg::Packet::empty();
    while opened_enc.receive_packet(&mut encoded_packet).is_ok() {
        encoded_packet.set_stream(output_stream_index);
        encoded_packet.rescale_ts(input_time_base, output_time_base);
        encoded_packet.write_interleaved(&mut octx)?;
    }

    // Write trailer
    octx.write_trailer()?;

    Ok(())
}