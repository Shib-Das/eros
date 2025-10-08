use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use std::{
    fs::File,
    path::{Path, PathBuf},
};
use tar::Builder;

/// Creates a compressed shrine archive.
///
/// The shrine is a `.tar.gz` file containing the media files and the database.
///
/// # Arguments
///
/// * `output_path` - The path where the shrine file will be created.
/// * `files_to_include` - A slice of paths to the media files to include.
/// * `db_path` - The path to the database file.
pub fn create_shrine(
    output_path: &Path,
    files_to_include: &[PathBuf],
    db_path: &Path,
) -> Result<()> {
    // 1. Create the output file for the shrine.
    let shrine_file = File::create(output_path)?;

    // 2. Wrap the file in a Gzip encoder for compression.
    let enc = GzEncoder::new(shrine_file, Compression::default());

    // 3. Initialize a `tar` builder with the Gzip encoder.
    let mut tar_builder = Builder::new(enc);

    // 4. Add the database file to the archive.
    tar_builder.append_path_with_name(db_path, "victim.db")?;

    // 5. Iterate through the media files and add them to the archive.
    for file_path in files_to_include {
        if file_path.is_file() {
            let file_name = file_path.file_name().unwrap_or_default();
            tar_builder.append_path_with_name(file_path, file_name)?;
        }
    }

    // 6. Finalize the archive.
    tar_builder.finish()?;

    Ok(())
}