use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ratatui::layout::Rect;

const ASCII_CHARS: [char; 11] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@', '$'];

/// Converts an image to ASCII art that fits within the given dimensions.
pub fn create_ascii_art(image: &DynamicImage, area: Rect) -> String {
    if area.width == 0 || area.height < 2 {
        return String::new();
    }

    // Adjust height to compensate for character aspect ratio in terminals
    let ascii_height = (area.height as f32 / 2.0).round() as u32;
    if ascii_height == 0 {
        return String::new();
    }

    let width = area.width as u32;
    let height = ascii_height;

    let resized_image = image.resize_exact(width, height, FilterType::Nearest);
    let mut ascii_art = String::new();

    for y in 0..resized_image.height() {
        for x in 0..resized_image.width() {
            let pixel = resized_image.get_pixel(x, y);
            let gray =
                (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
            let char_index = (gray as f32 / 255.0 * (ASCII_CHARS.len() - 1) as f32).round() as usize;
            ascii_art.push(ASCII_CHARS[char_index]);
        }
        ascii_art.push('\n');
    }

    ascii_art
}