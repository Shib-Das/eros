use eros::{
    rating::{Rating, RatingModel},
    tagger::{Device, TaggerModel},
};
use tokio::runtime::Runtime;

mod common;

fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    Runtime::new().unwrap().block_on(future)
}

#[test]
fn test_rating_model() {
    TaggerModel::init(Device::cpu()).unwrap();
    let mut model = run_async(RatingModel::new()).unwrap();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let rating = model.rate(&image).unwrap();
    // NOTE: The expected rating is Sfw because the procedurally generated test image is
    // a simple, neutral gray square, which should not be classified as NSFW.
    assert_eq!(rating, Rating::Sfw);
}