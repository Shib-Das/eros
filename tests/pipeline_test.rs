use eros::{
    pipeline::TaggingPipeline,
    tagger::{Device, TaggerModel},
};
use tokio::runtime::Runtime;

mod common;
use common::setup;

fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    Runtime::new().unwrap().block_on(future)
}

fn get_pipeline() -> TaggingPipeline {
    setup();
    TaggerModel::init(Device::cpu()).unwrap();
    run_async(TaggingPipeline::from_pretrained(
        "SmilingWolf/wd-swinv2-tagger-v3",
        Device::cpu(),
        None,
    ))
    .unwrap()
}

#[test]
fn test_from_pretrained_pipeline() {
    let pipeline = get_pipeline();
    assert_eq!(pipeline.threshold, 0.5);
}

#[test]
fn test_predict() {
    let mut pipeline = get_pipeline();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let result = pipeline.predict(image, None).unwrap();

    assert!(!result.rating.is_empty());
    assert!(!result.general.is_empty());
    assert!(result.character.is_empty()); // No characters in this image

    // Check that the tags are sorted by probability
    let mut sorted = result.general.clone();
    sorted.sort_by(|_, v1, _, v2| v2.partial_cmp(v1).unwrap());
    assert_eq!(result.general, sorted);
}

#[test]
fn test_predict_batch() {
    let mut pipeline = get_pipeline();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let results = pipeline.predict_batch(vec![&image, &image], None).unwrap();

    assert_eq!(results.len(), 2);
    let result1 = &results[0];
    let result2 = &results[1];

    // Check that the results for the two identical images are the same
    assert_eq!(result1.rating, result2.rating);
    assert_eq!(result1.general, result2.general);
    assert_eq!(result1.character, result2.character);

    // Check sorting for the first result
    let mut sorted = result1.general.clone();
    sorted.sort_by(|_, v1, _, v2| v2.partial_cmp(v1).unwrap());
    assert_eq!(result1.general, sorted);
}