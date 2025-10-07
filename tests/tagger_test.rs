use eros::{
    processor::{ImagePreprocessor, ImageProcessor},
    tagger::{Device, TaggerModel},
    tags::LabelTags,
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

#[test]
fn test_from_pretrained() {
    setup();
    TaggerModel::init(Device::cpu()).unwrap();
    assert!(
        run_async(TaggerModel::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).is_ok()
    );
}

#[test]
fn test_predict() {
    setup();
    TaggerModel::init(Device::cpu()).unwrap();
    let mut model =
        run_async(TaggerModel::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
    let processor =
        run_async(ImagePreprocessor::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3"))
            .unwrap();
    let tags = run_async(LabelTags::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let input_tensor = processor.process(&image).unwrap();

    let predictions = model.predict(input_tensor).unwrap();
    assert_eq!(predictions.len(), 1); // Batch size of 1
    assert_eq!(predictions[0].len(), tags.idx2tag().len()); // Number of tags
}

#[test]
fn test_predict_batch() {
    setup();
    TaggerModel::init(Device::cpu()).unwrap();
    let mut model =
        run_async(TaggerModel::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
    let processor =
        run_async(ImagePreprocessor::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3"))
            .unwrap();
    let tags = run_async(LabelTags::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
    let image = image::open("tests/assets/test_image.jpg").unwrap();
    let input_tensor = processor.process_batch(vec![&image, &image]).unwrap();

    let predictions = model.predict(input_tensor).unwrap();
    assert_eq!(predictions.len(), 2); // Batch size of 2
    assert_eq!(predictions[0].len(), tags.idx2tag().len());
    assert_eq!(predictions[1].len(), tags.idx2tag().len());
}