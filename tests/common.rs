use image::{ImageBuffer, Rgb};
use std::{fs, path::Path, sync::Once};

#[allow(dead_code)]
static SETUP: Once = Once::new();

#[allow(dead_code)]
fn generate_test_image(path: &Path) {
    let img_width = 100;
    let img_height = 100;
    let mut img = ImageBuffer::new(img_width, img_height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([128 as u8, 128 as u8, 128 as u8]); // A neutral gray color
    }
    img.save(path).unwrap();
}

#[allow(dead_code)]
pub fn setup() {
    SETUP.call_once(|| {
        let assets_dir = Path::new("tests/assets");
        if !assets_dir.exists() {
            fs::create_dir_all(assets_dir).unwrap();
        }

        let image_path = assets_dir.join("test_image.jpg");
        if !image_path.exists() {
            generate_test_image(&image_path);
        }
    });
}