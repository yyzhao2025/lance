// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

/// Options for optimizing all indices.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
    /// Number of existing index segments to merge for one column. Default: 1.
    ///
    /// In current vector optimize paths, `None` means Lance may either append a
    /// new segment or merge all existing segments, depending on whether a
    /// partition split is required.
    ///
    /// If `num_indices_to_merge` is `Some(N)`, the latest N existing segments
    /// together with any newly-built data will be merged into one segment.
    ///
    /// It is up to the caller to decide how many segments to merge / keep.
    /// Callers can find out how many committed segments exist by calling
    /// `Dataset::index_statistics`.
    ///
    /// A common usage pattern is to keep a large retained segment snapshot and
    /// periodically merge newer segments back into that snapshot.
    pub num_indices_to_merge: Option<usize>,

    /// the index names to optimize. If None, all indices will be optimized.
    pub index_names: Option<Vec<String>>,

    /// whether to retrain the whole index. Default: false.
    ///
    /// If true, the index will be retrained based on the current data,
    /// `num_indices_to_merge` will be ignored, and all indices will be merged into one.
    /// If false, the index will be optimized by merging `num_indices_to_merge` indices.
    ///
    /// This is useful when the data distribution has changed significantly,
    /// and we want to retrain the index to improve the search quality.
    /// This would be faster than re-create the index from scratch.
    ///
    /// NOTE: this option is only supported for v3 vector indices.
    pub retrain: bool,

    /// Transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed index jobs).
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl OptimizeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(num: usize) -> Self {
        Self {
            num_indices_to_merge: Some(num),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn append() -> Self {
        Self {
            num_indices_to_merge: Some(0),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn retrain() -> Self {
        Self {
            num_indices_to_merge: None,
            index_names: None,
            retrain: true,
            ..Default::default()
        }
    }

    pub fn num_indices_to_merge(mut self, num: Option<usize>) -> Self {
        self.num_indices_to_merge = num;
        self
    }

    pub fn index_names(mut self, names: Vec<String>) -> Self {
        self.index_names = Some(names);
        self
    }

    /// Set transaction properties to store in the commit manifest.
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }
}
