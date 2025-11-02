pub use crate::error::{ErosError, Result};
use ffmpeg_next as ffmpeg;
use image::imageops::FilterType;
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
                    let resized_img = img.resize(size.0, size.1, FilterType::Lanczos3);
                    resized_img.save(path)?;
                } else if VIDEO_EXTENSIONS.contains(&ext_lower.as_str()) {
                    let temp_output_path = path.with_extension("temp_resized.mp4");

                    let mut ictx = ffmpeg::format::input(path)?;
                    let mut octx = ffmpeg::format::output(&temp_output_path)?;

                    let in_stream_index = {
                        let in_stream = ictx.streams().best(ffmpeg::media::Type::Video).ok_or(ffmpeg::Error::StreamNotFound)?;
                        in_stream.index()
                    };
                    let mut out_stream = octx.add_stream(None)?;

                    let context_decoder = ffmpeg::codec::context::Context::from_parameters(ictx.stream(in_stream_index).unwrap().parameters())?;
                    let mut decoder = context_decoder.decoder().video()?;

                    out_stream.set_parameters(ictx.stream(in_stream_index).unwrap().parameters());

                    let (new_width, new_height) = {
                        let width = decoder.width();
                        let height = decoder.height();
                        let ratio = width as f32 / height as f32;
                        if ratio > 1.0 {
                            (size.0, (size.0 as f32 / ratio).round() as u32)
                        } else {
                            ((size.1 as f32 * ratio).round() as u32, size.1)
                        }
                    };
                    let mut scaler = ffmpeg::software::scaling::context::Context::get(
                        decoder.format(),
                        decoder.width(),
                        decoder.height(),
                        ffmpeg::format::Pixel::YUV420P,
                        new_width,
                        new_height,
                        ffmpeg::software::scaling::flag::Flags::LANCZOS,
                    )?;

                    let context_encoder = ffmpeg::codec::context::Context::from_parameters(out_stream.parameters())?;
                    let mut encoder = context_encoder.encoder().video()?;

                    octx.write_header()?;

                    for (stream, packet) in ictx.packets() {
                        if stream.index() == in_stream_index {
                            decoder.send_packet(&packet)?;
                            let mut decoded = ffmpeg::frame::Video::empty();
                            while decoder.receive_frame(&mut decoded).is_ok() {
                                let mut scaled = ffmpeg::frame::Video::empty();
                                scaler.run(&decoded, &mut scaled)?;
                                encoder.send_frame(&scaled)?;
                                let mut encoded = ffmpeg::Packet::empty();
                                while encoder.receive_packet(&mut encoded).is_ok() {
                                    encoded.write_interleaved(&mut octx)?;
                                }
                            }
                        } else {
                            packet.write_interleaved(&mut octx)?;
                        }
                    }

                    octx.write_trailer()?;
                    fs::remove_file(path)?;
                    fs::rename(&temp_output_path, path.with_extension("mp4"))?;
                }
            }
        }
    }
    Ok(())
}
