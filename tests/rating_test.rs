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
    assert_eq!(rating, Rating::Sfw);
}