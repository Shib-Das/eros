use eros::file::download_file;
use std::{
    fs,
    path::Path,
    sync::Once,
};
use tokio::runtime::Runtime;

static SETUP: Once = Once::new();

pub fn setup() {
    SETUP.call_once(|| {
        let assets_dir = Path::new("tests/assets");
        if !assets_dir.exists() {
            fs::create_dir_all(assets_dir).unwrap();
        }

        let image_path = assets_dir.join("test_image.jpg");
        if !image_path.exists() {
            Runtime::new()
                .unwrap()
                .block_on(download_file(
                    "https://drive.usercontent.google.com/download?id=13Jos4-EHhcXSIhoQ6TZluB65bpZiwRpK&export=download&authuser=3&confirm=t&uuid=d3c10b48-4408-43ac-bf5e-8cf83a730258&at=AN8xHooYiPV77OKFoIdL-i3XuJLK:1759283495654",
                    &image_path,
                ))
                .expect("Failed to download test image");
        }

        let video_path = assets_dir.join("test_video.mp4");
        if !video_path.exists() {
            Runtime::new()
                .unwrap()
                .block_on(download_file(
                    "https://drive.usercontent.google.com/download?id=17BgSyqj8mMPIJSnSTN52CxuvcdOBzOhh&export=download&authuser=0",
                    &video_path,
                ))
                .expect("Failed to download test video");
        }
    });
}