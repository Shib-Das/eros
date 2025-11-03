pub use crate::error::{ErosError, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "webp"];

pub fn suggest_media_directories(start_path: &Path) -> Result<Vec<PathBuf>> {
    let mut media_dirs = Vec::new();

    for entry in WalkDir::new(start_path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                if IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str())
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
                }
            }
        }
    }
    Ok(())
}
