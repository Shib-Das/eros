use crate::error::TaggerError;
use crate::file::TagCSVFile;
use anyhow::Result;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Each record in the CSV file
#[derive(Debug, Deserialize, Clone)]
pub struct Tag {
    tag_id: i32,
    name: String,
    category: TagCategory,
    count: i32,
}

/// Tag category
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub enum TagCategory {
    #[serde(rename = "0")]
    General,
    #[serde(rename = "1")]
    Artist,
    #[serde(rename = "3")]
    Copyright,
    #[serde(rename = "4")]
    Character,
    #[serde(rename = "5")]
    Meta,
    #[serde(rename = "9")]
    Rating,
}

impl Tag {
    pub fn category(&self) -> TagCategory {
        self.category.clone()
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn tag_id(&self) -> i32 {
        self.tag_id
    }

    pub fn count(&self) -> i32 {
        self.count
    }
}

/// The tags in the CSV file
#[derive(Debug, Clone)]
pub struct LabelTags {
    label2tag: HashMap<String, Tag>,
    idx2tag: HashMap<usize, Tag>,
    embeddings: Option<Array2<f32>>,
}

impl LabelTags {
    /// Load from the local CSV file
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, TaggerError> {
        let mut reader =
            csv::Reader::from_path(csv_path.as_ref()).map_err(|e| TaggerError::Tag(e.to_string()))?;
        let headers = reader
            .headers()
            .map_err(|e| TaggerError::Tag(e.to_string()))?
            .clone();

        let embedding_cols: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter(|(_, h)| h.starts_with("embedding__"))
            .map(|(i, _)| i)
            .collect();
        let has_embeddings = !embedding_cols.is_empty();

        let mut label2tag = HashMap::new();
        let mut idx2tag = HashMap::new();
        let mut embeddings_vec = Vec::new();

        let records: Vec<_> = reader
            .records()
            .collect::<Result<_, _>>()
            .map_err(|e| TaggerError::Tag(e.to_string()))?;
        for (i, record) in records.iter().enumerate() {
            let tag: Tag = record
                .deserialize(Some(&headers))
                .map_err(|e| TaggerError::Tag(e.to_string()))?;

            label2tag.insert(tag.name.clone(), tag.clone());
            idx2tag.insert(i, tag);

            if has_embeddings {
                let mut embedding_row = Vec::new();
                for &col_idx in &embedding_cols {
                    let val: f32 = record[col_idx]
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| TaggerError::Tag(e.to_string()))?;
                    embedding_row.push(val);
                }
                embeddings_vec.push(embedding_row);
            }
        }

        let embeddings = if has_embeddings {
            let rows = embeddings_vec.len();
            let cols = if rows > 0 {
                embeddings_vec[0].len()
            } else {
                0
            };
            if cols > 0 {
                let flat_embeddings: Vec<f32> = embeddings_vec.into_iter().flatten().collect();
                Some(
                    Array2::from_shape_vec((rows, cols), flat_embeddings).map_err(|e| {
                        TaggerError::Tag(format!("Failed to create embedding array: {}", e))
                    })?,
                )
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            label2tag,
            idx2tag,
            embeddings,
        })
    }

    pub async fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let csv_path = TagCSVFile::new(repo_id).get().await?;
        Self::load(csv_path)
    }

    /// Create pairs of tag and probability with given tensor
    pub fn create_probality_pairs(
        &self,
        tensor: Vec<Vec<f32>>,
    ) -> Result<Vec<IndexMap<String, f32>>, TaggerError> {
        tensor
            .iter() // batch
            .map(|probs| {
                let probs_vec = if let Some(embeddings) = &self.embeddings {
                    if probs.len() != embeddings.shape()[1] {
                        return Err(TaggerError::Tag(format!(
                            "Prediction feature size ({}) mismatch with embedding dimension ({})",
                            probs.len(),
                            embeddings.shape()[1]
                        )));
                    }
                    let pred_array = Array1::from_vec(probs.clone());
                    let similarities = embeddings.dot(&pred_array);
                    similarities.to_vec()
                } else {
                    if probs.len() != self.idx2tag.len() {
                        return Err(TaggerError::Tag(
                            "Tags and probabilities length mismatch".to_string(),
                        ));
                    }
                    probs.to_vec()
                };

                Ok(probs_vec
                    .iter()
                    .enumerate()
                    .map(|(idx, prob)| (self.idx2tag.get(&idx).unwrap().name(), *prob))
                    .collect::<IndexMap<String, f32>>())
            })
            .collect()
    }

    pub fn label2tag(&self) -> &HashMap<String, Tag> {
        &self.label2tag
    }

    pub fn idx2tag(&self) -> &HashMap<usize, Tag> {
        &self.idx2tag
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tokio::runtime::Runtime;

    fn run_async<F, T>(future: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        Runtime::new().unwrap().block_on(future)
    }

    #[test]
    fn test_from_pretrained_tags() {
        let tags = run_async(LabelTags::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
        assert!(!tags.label2tag().is_empty());
        assert!(!tags.idx2tag().is_empty());
    }

    #[test]
    fn test_create_probability_pairs() {
        let tags = run_async(LabelTags::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
        let num_tags = tags.idx2tag().len();
        let probabilities = vec![vec![0.1; num_tags], vec![0.2; num_tags]];

        let pairs = tags.create_probality_pairs(probabilities).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].len(), num_tags);
        assert_eq!(pairs[1].len(), num_tags);
        assert_eq!(*pairs[0].get("1girl").unwrap(), 0.1);
        assert_eq!(*pairs[1].get("1girl").unwrap(), 0.2);
    }

    #[test]
    fn test_create_probability_pairs_mismatch() {
        let tags = run_async(LabelTags::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3")).unwrap();
        let num_tags = tags.idx2tag().len();
        let probabilities = vec![vec![0.1; num_tags + 1]]; // Mismatched length

        let result = tags.create_probality_pairs(probabilities);
        assert!(result.is_err());
        match result.unwrap_err() {
            TaggerError::Tag(msg) => {
                assert_eq!(msg, "Tags and probabilities length mismatch");
            }
            _ => panic!("Expected TaggerError::Tag"),
        }
    }
}