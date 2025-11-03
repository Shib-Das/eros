use anyhow::Result;
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

fn remux(from: &Path, to: &Path) -> Result<(), ffmpeg::Error> {
    let mut ictx = ffmpeg::format::input(&from)?;
    let mut octx = ffmpeg::format::output_as(&to, "mp4")?;

    octx.set_metadata(Default::default());

    let mut stream_mapping = vec![0; ictx.nb_streams() as usize];
    for in_stream in ictx.streams() {
        let mut out_stream = octx.add_stream(None)?;
        out_stream.set_parameters(in_stream.parameters());
        out_stream.set_metadata(Default::default());
        stream_mapping[in_stream.index()] = out_stream.index();
    }

    octx.write_header()?;

    for (stream, mut packet) in ictx.packets() {
        let out_stream_index = stream_mapping[stream.index()];
        let out_stream = octx.stream(out_stream_index).unwrap();

        packet.rescale_ts(stream.time_base(), out_stream.time_base());
        packet.set_stream(out_stream_index);
        packet.write_interleaved(&mut octx)?;
    }

    octx.write_trailer()?;
    Ok(())
}

use std::process::{Command, Stdio};

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

fn resize_video(from: &Path, to: &Path, size: (u32, u32)) -> anyhow::Result<()> {
    let (width, height) = size;
    let vf_param = format!("scale={}:{}", width, height);

    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(from)
        .arg("-vf")
        .arg(&vf_param)
        .arg("-c:a")
        .arg("copy")
        .arg(to)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;

    if !status.success() {
        return Err(anyhow::anyhow!("ffmpeg failed to resize video"));
    }

    Ok(())
}