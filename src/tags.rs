use crate::file::TagCSVFile;
use anyhow::{Context, Result};
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
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(csv_path.as_ref())
            .with_context(|| format!("Failed to read CSV file at {:?}", csv_path.as_ref()))?;
        let headers = reader.headers()?.clone();
        let records: Vec<_> = reader.records().collect::<Result<_, _>>()?;

        let embedding_cols: Vec<_> = headers
            .iter()
            .enumerate()
            .filter(|(_, h)| h.starts_with("embedding__"))
            .map(|(i, _)| i)
            .collect();

        let mut label2tag = HashMap::with_capacity(records.len());
        let mut idx2tag = HashMap::with_capacity(records.len());
        let mut embeddings_vec = Vec::with_capacity(records.len());

        for (i, record) in records.iter().enumerate() {
            let tag: Tag = record
                .deserialize(Some(&headers))
                .context("Failed to deserialize tag record")?;
            label2tag.insert(tag.name.clone(), tag.clone());
            idx2tag.insert(i, tag);

            if !embedding_cols.is_empty() {
                let embedding_row: Result<Vec<f32>, _> = embedding_cols
                    .iter()
                    .map(|&col_idx| {
                        record[col_idx]
                            .parse()
                            .with_context(|| format!("Failed to parse embedding value at column {}", col_idx))
                    })
                    .collect();
                embeddings_vec.push(embedding_row?);
            }
        }

        let embeddings = if !embeddings_vec.is_empty() {
            let rows = embeddings_vec.len();
            let cols = embeddings_vec[0].len();
            let flat_embeddings: Vec<f32> = embeddings_vec.into_iter().flatten().collect();
            Some(
                Array2::from_shape_vec((rows, cols), flat_embeddings)
                    .context("Failed to create embedding array")?,
            )
        } else {
            None
        };

        Ok(Self {
            label2tag,
            idx2tag,
            embeddings,
        })
    }

    pub async fn from_pretrained(repo_id: &str) -> Result<Self> {
        let csv_path = TagCSVFile::new(repo_id).get().await?;
        Self::load(csv_path)
    }

    /// Create pairs of tag and probability with given tensor
    pub fn create_probality_pairs(
        &self,
        tensor: Vec<Vec<f32>>,
    ) -> Result<Vec<IndexMap<String, f32>>> {
        tensor
            .into_iter()
            .map(|probs| {
                let probs_vec = self.get_probs_vec(probs)?;
                Ok(probs_vec
                    .into_iter()
                    .enumerate()
                    .map(|(idx, prob)| (self.idx2tag[&idx].name(), prob))
                    .collect())
            })
            .collect()
    }

    fn get_probs_vec(&self, probs: Vec<f32>) -> Result<Vec<f32>> {
        if let Some(embeddings) = &self.embeddings {
            anyhow::ensure!(
                probs.len() == embeddings.shape()[1],
                "Prediction feature size ({}) mismatch with embedding dimension ({})",
                probs.len(),
                embeddings.shape()[1]
            );
            let pred_array = Array1::from_vec(probs);
            Ok(embeddings.dot(&pred_array).to_vec())
        } else {
            anyhow::ensure!(
                probs.len() == self.idx2tag.len(),
                "Tags and probabilities length mismatch"
            );
            Ok(probs)
        }
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
        assert_eq!(
            result.unwrap_err().to_string(),
            "Tags and probabilities length mismatch"
        );
    }
}