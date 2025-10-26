use eros::processor::{ImagePreprocessor, ImageProcessor};
use image::{Rgb, RgbImage};
use ndarray::s;
use tokio::runtime::Runtime;

mod common;
use common::setup;

fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    Runtime::new().unwrap().block_on(future)
}

#[test]
fn test_process_image() {
    setup();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let processor = ImagePreprocessor::new(
        448,
        448,
        vec![0.48145466, 0.4578275, 0.40821073],
        vec![0.26862954, 0.26130258, 0.27577711],
        true, // Test NHWC layout
    );
    let tensor = processor.process(&image).unwrap();

    // Check the shape of the output tensor
    assert_eq!(tensor.shape(), &[1, 448, 448, 3]);

    // Check that the tensor is not all zeros
    assert!(tensor.iter().any(|&x| x != 0.0));
}

#[test]
fn test_process_batch() {
    setup();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let processor = ImagePreprocessor::new(
        448,
        448,
        vec![0.48145466, 0.4578275, 0.40821073],
        vec![0.26862954, 0.26130258, 0.27577711],
        true, // Test NHWC layout
    );
    let batch_tensor = processor.process_batch(vec![&image, &image]).unwrap();

    // Check the shape of the output tensor
    assert_eq!(batch_tensor.shape(), &[2, 448, 448, 3]);

    // Check that the two images in the batch are identical
    let image1 = batch_tensor.slice(s![0, .., .., ..]);
    let image2 = batch_tensor.slice(s![1, .., .., ..]);
    assert_eq!(image1, image2);
}

#[test]
fn test_from_pretrained_processor() {
    setup();
    let processor = run_async(ImagePreprocessor::from_pretrained(
        "SmilingWolf/wd-swinv2-tagger-v3",
    ))
    .unwrap();
    assert_eq!(processor.height, 448);
    assert_eq!(processor.width, 448);
    assert_eq!(processor.bgr, true); // This model uses the fallback NHWC layout
    assert_eq!(processor.mean, vec![0.48145466, 0.4578275, 0.40821073]);
}

#[test]
fn test_aspect_ratio_preservation() {
    setup();
    // Create a wide, non-square image (800x200) filled with red.
    let wide_image = RgbImage::from_pixel(800, 200, Rgb([255, 0, 0]));
    let dynamic_wide_image = image::DynamicImage::ImageRgb8(wide_image);

    // Use simple normalization values for easy testing.
    let mean = vec![0.5, 0.5, 0.5];
    let std = vec![0.5, 0.5, 0.5];

    let processor = ImagePreprocessor::new(
        448,
        448,
        mean.clone(),
        std.clone(),
        false, // Use NCHW for simplicity.
    );

    let tensor = processor.process(&dynamic_wide_image).unwrap();

    // The 800x200 image should be resized to 448x112 to fit the 448x448 target.
    // This means there will be vertical padding.
    // The height of the top padding = (448 - 112) / 2 = 168 pixels.

    // The padding color is gray [128, 128, 128].
    // Normalized padding value: (128/255 - 0.5) / 0.5 ≈ 0.00392
    let norm_pad_val = (128.0 / 255.0 - mean[0]) / std[0];

    // The first row of the tensor should be all padding.
    let top_row_r = tensor.slice(s![0, 0, 0, ..]);
    assert!(top_row_r.iter().all(|&v| (v - norm_pad_val).abs() < 1e-5));

    // The center of the image should be red [255, 0, 0].
    // Normalized red channel: (255/255 - 0.5) / 0.5 = 1.0
    // Normalized green/blue channels: (0/255 - 0.5) / 0.5 = -1.0
    let norm_r = (255.0 / 255.0 - mean[0]) / std[0];
    let norm_g = (0.0 / 255.0 - mean[1]) / std[1];

    let center_pixel_r = tensor[[0, 0, 224, 224]];
    let center_pixel_g = tensor[[0, 1, 224, 224]];

    assert!((center_pixel_r - norm_r).abs() < 1e-5);
    assert!((center_pixel_g - norm_g).abs() < 1e-5);
}