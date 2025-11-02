pub use crate::error::{ErosError, Result};
use ffmpeg_next as ffmpeg;
use image::{codecs::png::PngEncoder, ImageEncoder};
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

use rayon::prelude::*;
use tempfile::{tempdir, Builder};

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
                    let resized_img = img.thumbnail(size.0, size.1);
                    resized_img.save(path)?;
                } else if VIDEO_EXTENSIONS.contains(&ext_lower.as_str()) {
                    // Step 1: Analyze Video
                    let mut ictx = ffmpeg::format::input(&path)?;

                    let best_video_stream = ictx
                        .streams()
                        .best(ffmpeg::media::Type::Video)
                        .ok_or(ffmpeg::Error::StreamNotFound)?;
                    let video_stream_index = best_video_stream.index();
                    let original_frame_rate = best_video_stream.avg_frame_rate();
                    let original_time_base = best_video_stream.time_base();

                    let best_audio_stream = ictx.streams().best(ffmpeg::media::Type::Audio);
                    let audio_stream_index = best_audio_stream.as_ref().map(|s| s.index());

                    // Step 2: Separate Frames and Audio
                    let temp_dir = tempdir()?;
                    let temp_audio_path = Builder::new().suffix(".m4a").tempfile()?.into_temp_path();

                    let mut octx_audio = ffmpeg::format::output(&temp_audio_path)?;

                    if let Some(best_audio_stream) = best_audio_stream {
                        let mut out_audio_stream = octx_audio.add_stream(None)?;
                        out_audio_stream.set_parameters(best_audio_stream.parameters());
                    }
                    octx_audio.write_header()?;

                    let video_stream = ictx
                        .streams()
                        .best(ffmpeg::media::Type::Video)
                        .ok_or(ffmpeg::Error::StreamNotFound)?;
                    let context_decoder =
                        ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
                    let mut decoder = context_decoder.decoder().video()?;
                    let mut scaler = ffmpeg::software::scaling::context::Context::get(
                        decoder.format(),
                        decoder.width(),
                        decoder.height(),
                        ffmpeg::format::Pixel::RGB24,
                        decoder.width(),
                        decoder.height(),
                        ffmpeg::software::scaling::flag::Flags::LANCZOS,
                    )?;

                    let mut frame_count = 0;
                    for (stream, packet) in ictx.packets() {
                        if stream.index() == video_stream_index {
                            decoder.send_packet(&packet)?;
                            let mut decoded = ffmpeg::frame::Video::empty();
                            while decoder.receive_frame(&mut decoded).is_ok() {
                                let mut rgb_frame = ffmpeg::frame::Video::empty();
                                scaler.run(&decoded, &mut rgb_frame)?;
                                let frame_path = temp_dir
                                    .path()
                                    .join(format!("frame_{:08}.png", frame_count));
                                let file = fs::File::create(frame_path)?;
                                let writer = std::io::BufWriter::new(file);
                                let encoder = PngEncoder::new(writer);
                                encoder.write_image(
                                    rgb_frame.data(0),
                                    rgb_frame.width(),
                                    rgb_frame.height(),
                                    image::ColorType::Rgb8.into(),
                                )?;
                                frame_count += 1;
                            }
                        } else if Some(stream.index()) == audio_stream_index {
                            packet.write_interleaved(&mut octx_audio)?;
                        }
                    }
                    octx_audio.write_trailer()?;

                    // Step 3: Resize All Frames
                    let frame_files: Vec<_> = WalkDir::new(temp_dir.path())
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_type().is_file())
                        .map(|e| e.path().to_path_buf())
                        .collect();

                    frame_files.par_iter().try_for_each(|frame_path| -> Result<()> {
                        let img = image::open(&frame_path)?;
                        let resized_img = img.thumbnail(size.0, size.1);
                        resized_img.save(&frame_path)?;
                        Ok(())
                    })?;

                    // Step 4: Re-assemble Video and Audio
                    let temp_video_final_path = path.with_extension("temp_final.mp4");
                    let mut octx_final = ffmpeg::format::output(&temp_video_final_path)?;

                    let first_frame_path = temp_dir.path().join("frame_00000000.png");
                    let first_frame = image::open(&first_frame_path)?;
                    let (new_width, new_height) = (first_frame.width(), first_frame.height());

                    let h264_codec = ffmpeg::encoder::find(ffmpeg::codec::Id::H264)
                        .ok_or_else(|| ErosError::Optimizer("H264 encoder not found".to_string()))?;

                    let mut encoder_ctx =
                        ffmpeg::codec::context::Context::new_with_codec(h264_codec)
                            .encoder()
                            .video()?;

                    encoder_ctx.set_height(new_height);
                    encoder_ctx.set_width(new_width);
                    encoder_ctx.set_format(ffmpeg::format::Pixel::YUV420P);
                    encoder_ctx.set_frame_rate(Some(original_frame_rate));
                    encoder_ctx.set_time_base(original_time_base);

                    let mut encoder = encoder_ctx.open_as(h264_codec)?;

                    let out_video_stream_index;
                    {
                        let mut out_stream = octx_final.add_stream(None)?;
                        out_video_stream_index = out_stream.index();
                        out_stream.set_parameters(&encoder);
                    }

                    let out_audio_stream_index = if let Ok(ictx_audio) = ffmpeg::format::input(&temp_audio_path) {
                        if let Some(in_stream) = ictx_audio.streams().best(ffmpeg::media::Type::Audio) {
                            let mut out_stream = octx_final.add_stream(None)?;
                            out_stream.set_parameters(in_stream.parameters());
                            Some(out_stream.index())
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    octx_final.write_header().map_err(|e| ErosError::Optimizer(e.to_string()))?;

                    let out_video_stream_time_base = octx_final.stream(out_video_stream_index).unwrap().time_base();
                    // Encoding Loop
                    for i in 0..frame_files.len() {
                        let frame_path = temp_dir.path().join(format!("frame_{:08}.png", i));
                        let img = image::open(&frame_path)?.to_rgb8();
                        let mut frame = ffmpeg::frame::Video::new(
                            ffmpeg::format::Pixel::RGB24,
                            new_width,
                            new_height,
                        );
                        frame.data_mut(0)[..img.len()].copy_from_slice(&img);

                        let mut yuv_frame = ffmpeg::frame::Video::new(
                            ffmpeg::format::Pixel::YUV420P,
                            new_width,
                            new_height,
                        );
                        let mut scaler = ffmpeg::software::scaling::context::Context::get(
                            ffmpeg::format::Pixel::RGB24,
                            new_width,
                            new_height,
                            ffmpeg::format::Pixel::YUV420P,
                            new_width,
                            new_height,
                            ffmpeg::software::scaling::flag::Flags::LANCZOS,
                        )?;
                        scaler.run(&frame, &mut yuv_frame)?;
                        yuv_frame.set_pts(Some(i as i64));

                        encoder.send_frame(&yuv_frame)?;
                        let mut encoded = ffmpeg::Packet::empty();
                        while encoder.receive_packet(&mut encoded).is_ok() {
                            encoded.set_stream(out_video_stream_index);
                            encoded.rescale_ts(original_time_base, out_video_stream_time_base);
                            encoded.write_interleaved(&mut octx_final).map_err(|e| ErosError::Optimizer(e.to_string()))?;
                        }
                    }
                    encoder.send_eof()?;
                    let mut encoded = ffmpeg::Packet::empty();
                    while encoder.receive_packet(&mut encoded).is_ok() {
                        encoded.set_stream(out_video_stream_index);
                        encoded.rescale_ts(original_time_base, out_video_stream_time_base);
                        encoded.write_interleaved(&mut octx_final).map_err(|e| ErosError::Optimizer(e.to_string()))?;
                    }

                    // Audio Remux Loop
                    if let (Ok(mut ictx_audio), Some(audio_index)) = (ffmpeg::format::input(&temp_audio_path), out_audio_stream_index) {
                        let out_audio_stream_time_base = octx_final.stream(audio_index).unwrap().time_base();
                        for (stream, mut packet) in ictx_audio.packets() {
                            packet.rescale_ts(stream.time_base(), out_audio_stream_time_base);
                            packet.set_stream(audio_index);
                            packet.write_interleaved(&mut octx_final).map_err(|e| ErosError::Optimizer(e.to_string()))?;
                        }
                    }

                    octx_final.write_trailer().map_err(|e| ErosError::Optimizer(e.to_string()))?;

                    fs::remove_file(&path)?;
                    fs::rename(&temp_video_final_path, &path)?;
                }
            }
        }
    }
    Ok(())
}
