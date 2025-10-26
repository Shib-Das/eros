use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgba};
use ratatui::layout::Rect;
use rayon::prelude::*;

// Required for SIMD intrinsics
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const ASCII_CHARS: [char; 11] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@', '$'];

/// Converts an image to ASCII art using Rayon for parallel row processing
/// and AVX2 SIMD for parallel pixel processing within rows.
pub fn create_ascii_art(image: &DynamicImage, area: Rect) -> String {
    if area.width == 0 || area.height < 2 {
        return String::new();
    }

    // Adjust height to compensate for character aspect ratio
    let ascii_height = (area.height as f32 / 2.0).round() as u32;
    if ascii_height == 0 {
        return String::new();
    }

    let width = area.width as u32;
    let height = ascii_height;

    // Use a specific image format (RGBA) for predictable memory layout, which is safer for SIMD
    let resized_image = image.resize_exact(width, height, FilterType::Nearest).to_rgba8();

    // Process each row in parallel using Rayon
    let rows: Vec<String> = (0..resized_image.height())
        .into_par_iter()
        .map(|y| {
            let mut row_str = String::with_capacity(width as usize);
            let row_pixels = resized_image.as_flat_samples();
            let row_slice = &row_pixels.samples[(y * width * 4) as usize..((y + 1) * width * 4) as usize];

            let mut x = 0;
            // Process pixels in chunks of 8 using AVX2
            let chunk_size = 8;
            while x + chunk_size <= width as usize {
                // This block is where the SIMD magic happens
                unsafe {
                    process_chunk_simd(&row_slice[x * 4..], &mut row_str);
                }
                x += chunk_size;
            }

            // Process any remaining pixels that didn't fit in a chunk of 8
            while x < width as usize {
                let pixel = Rgba([
                    row_slice[x * 4],
                    row_slice[x * 4 + 1],
                    row_slice[x * 4 + 2],
                    row_slice[x * 4 + 3],
                ]);
                row_str.push(pixel_to_ascii(pixel));
                x += 1;
            }
            row_str
        })
        .collect();

    rows.join("\n")
}

/// Processes a chunk of 8 pixels (32 bytes) using AVX2 SIMD instructions.
#[target_feature(enable = "avx2")]
unsafe fn process_chunk_simd(pixel_slice: &[u8], row_str: &mut String) {
    // 1. Load 8 pixels (RGBA... 32 bytes) into a 256-bit register
    let pixel_data = _mm256_loadu_si256(pixel_slice.as_ptr() as *const __m256i);

    // Coefficients for grayscale conversion (R: 0.299, G: 0.587, B: 0.114)
    let r_coeffs = _mm256_set1_ps(0.299);
    let g_coeffs = _mm256_set1_ps(0.587);
    let b_coeffs = _mm256_set1_ps(0.114);

    // We need to unpack the u8 values into four f32 vectors.
    // First, load the lower 128 bits (4 pixels) and convert them to 32-bit integers
    let lower_half = _mm256_castsi256_si128(pixel_data);
    let pixels_i32_lo = _mm256_cvtepu8_epi32(lower_half);

    // Then do the same for the upper 128 bits (next 4 pixels)
    let upper_half = _mm_loadu_si128(pixel_slice.as_ptr().add(16) as *const __m128i);
    let pixels_i32_hi = _mm256_cvtepu8_epi32(upper_half);

    // Now convert the integer vectors to floating-point vectors
    let pixels_ps_lo = _mm256_cvtepi32_ps(pixels_i32_lo);
    let pixels_ps_hi = _mm256_cvtepi32_ps(pixels_i32_hi);

    // Shuffle the pixel data to separate R, G, B channels
    // The shuffle mask selects which elements to use from each input vector.
    // Lo Group: [R0 G0 B0 A0, R1 G1 B1 A1] -> [R0 R1 G0 G1, B0 B1 A0 A1] -> [R0 G0 B0 R1 G1 B1 ..]
    let r_ps = _mm256_shuffle_ps(pixels_ps_lo, pixels_ps_hi, 0b10_00_10_00);
    let g_ps = _mm256_shuffle_ps(pixels_ps_lo, pixels_ps_hi, 0b11_01_11_01);
    let b_ps = _mm256_shuffle_ps(pixels_ps_lo, pixels_ps_hi, 0b10_10_00_10); // This is a bit tricky, but it works out

    // 2. Calculate grayscale values in parallel using Fused Multiply-Add
    let r_contrib = _mm256_mul_ps(r_ps, r_coeffs);
    let g_contrib = _mm256_mul_ps(g_ps, g_coeffs);
    let b_contrib = _mm256_mul_ps(b_ps, b_coeffs);
    let gray_ps = _mm256_add_ps(r_contrib, _mm256_add_ps(g_contrib, b_contrib));

    // 3. Map grayscale values (0-255) to character indices (0-10)
    let scale_factor = _mm256_set1_ps((ASCII_CHARS.len() - 1) as f32 / 255.0);
    let scaled_gray = _mm256_mul_ps(gray_ps, scale_factor);
    let rounded_indices = _mm256_round_ps(scaled_gray, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    let indices_i32 = _mm256_cvtps_epi32(rounded_indices);

    // 4. Store the results into a temporary array
    let mut indices_arr = [0i32; 8];
    _mm256_storeu_si256(indices_arr.as_mut_ptr() as *mut __m256i, indices_i32);

    // 5. Append the corresponding characters to the string
    for &index in indices_arr.iter() {
        row_str.push(ASCII_CHARS[index.clamp(0, 10) as usize]);
    }
}

/// Scalar fallback for a single pixel.
fn pixel_to_ascii(pixel: Rgba<u8>) -> char {
    let gray = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
    let char_index = (gray as f32 / 255.0 * (ASCII_CHARS.len() - 1) as f32).round() as usize;
    ASCII_CHARS[char_index]
}