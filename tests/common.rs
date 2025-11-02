use ffmpeg_next as ffmpeg;
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
fn generate_test_video(path: &Path) {
    ffmpeg::init().unwrap();

    let mut octx = ffmpeg::format::output(&path).unwrap();
    let global_header = octx
        .format()
        .flags()
        .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

    let mut encoder;
    let stream_time_base;
    let stream_index;

    let width = 320;
    let height = 240;
    let framerate = 30;

    {
        let mut stream = octx.add_stream(ffmpeg::codec::Id::MPEG4).unwrap();
        stream.set_time_base((1, 600)); // Set a compliant timebase for the stream
        let encoder_ctx =
            ffmpeg::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let mut video_encoder = encoder_ctx.encoder().video().unwrap();

        video_encoder.set_width(width);
        video_encoder.set_height(height);
        video_encoder.set_format(ffmpeg::format::Pixel::YUV420P);
        video_encoder.set_time_base((1, 30)); // Encoder timebase can be different
        if global_header {
            video_encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }

        let opened_encoder = video_encoder
            .open_as(ffmpeg::codec::Id::MPEG4)
            .unwrap();
        stream.set_parameters(&opened_encoder);

        stream_time_base = stream.time_base();
        stream_index = stream.index();
        encoder = opened_encoder;
    }

    octx.write_header().unwrap();

    let mut frame = ffmpeg::frame::Video::new(encoder.format(), width, height);
    for i in 0..framerate * 2 {
        // 2 seconds of video
        frame.set_pts(Some(i as i64));
        // Simple pattern: a bar moving from left to right
        for y in 0..height {
            for x in 0..width {
                let color = if x > i as u32 * 4 && x < i as u32 * 4 + 20 {
                    255
                } else {
                    0
                };
                unsafe {
                    *frame
                        .data_mut(0)
                        .as_mut_ptr()
                        .offset((y * frame.stride(0) as u32 + x) as isize) = color;
                }
            }
        }
        encoder.send_frame(&frame).unwrap();
        let mut packet = ffmpeg_next::Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.rescale_ts(encoder.time_base(), stream_time_base);
            packet.set_stream(stream_index);
            packet.write_interleaved(&mut octx).unwrap();
        }
    }

    encoder.send_eof().unwrap();
    let mut packet = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut packet).is_ok() {
        packet.rescale_ts(encoder.time_base(), stream_time_base);
        packet.set_stream(stream_index);
        packet.write_interleaved(&mut octx).unwrap();
    }

    octx.write_trailer().unwrap();
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

        let video_path = assets_dir.join("test_video.mp4");
        if !video_path.exists() {
            generate_test_video(&video_path);
        }
    });
}