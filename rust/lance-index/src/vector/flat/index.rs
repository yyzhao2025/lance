// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID_FIELD, Result};
use lance_file::previous::reader::FileReader as PreviousFileReader;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::{
    metrics::MetricsCollector,
    prefilter::PreFilter,
    vector::{
        DIST_COL, Query,
        bq::storage::RabitQuantizationStorage,
        graph::OrderedNode,
        quantizer::{Quantization, QuantizationType, Quantizer, QuantizerMetadata},
        storage::{DistCalculator, VectorStore},
        v3::subindex::IvfSubIndex,
    },
};

use super::storage::{FLAT_COLUMN, FlatBinStorage, FlatFloatStorage};

/// A Flat index is any index that stores no metadata, and
/// during query, it simply scans over the storage and returns the top k results
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct FlatIndex {}

use std::sync::LazyLock;

static ANN_SEARCH_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ])
    .into()
});

#[derive(Default)]
pub struct FlatQueryParams {
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    dist_q_c: f32,
}

impl From<&Query> for FlatQueryParams {
    fn from(q: &Query) -> Self {
        Self {
            lower_bound: q.lower_bound,
            upper_bound: q.upper_bound,
            dist_q_c: q.dist_q_c,
        }
    }
}

impl IvfSubIndex for FlatIndex {
    type QueryParams = FlatQueryParams;
    type BuildParams = ();

    fn name() -> &'static str {
        "FLAT"
    }

    fn metadata_key() -> &'static str {
        "lance:flat"
    }

    fn schema() -> arrow_schema::SchemaRef {
        Schema::new(vec![Field::new("__flat_marker", DataType::UInt64, false)]).into()
    }

    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let is_range_query = params.lower_bound.is_some() || params.upper_bound.is_some();
        let row_ids = storage.row_ids();
        let mut res = BinaryHeap::with_capacity(k);

        if !is_range_query
            && prefilter.is_empty()
            && let Some(rq_storage) = storage.as_any().downcast_ref::<RabitQuantizationStorage>()
        {
            let dist_calc = rq_storage.dist_calculator(query, params.dist_q_c);
            let (results, stats) =
                dist_calc.search_topk_unfiltered_with_stats(rq_storage.row_ids_slice(), k);
            metrics.record_comparisons(stats.searched_rows);
            metrics.record_pruned_rows(stats.pruned_rows);
            let (row_ids, dists): (Vec<_>, Vec<_>) =
                results.into_iter().map(|r| (r.id, r.dist.0)).unzip();
            let (row_ids, dists) = (UInt64Array::from(row_ids), Float32Array::from(dists));

            return Ok(RecordBatch::try_new(
                ANN_SEARCH_SCHEMA.clone(),
                vec![Arc::new(dists), Arc::new(row_ids)],
            )?);
        }

        let dist_calc = storage.dist_calculator(query, params.dist_q_c);

        match prefilter.is_empty() {
            true => {
                metrics.record_comparisons(storage.len());
                let dists = dist_calc.distance_all(k);

                if is_range_query {
                    let lower_bound = params.lower_bound.unwrap_or(f32::MIN).into();
                    let upper_bound = params.upper_bound.unwrap_or(f32::MAX).into();

                    for (&row_id, dist) in row_ids.zip(dists) {
                        let dist = dist.into();
                        if dist < lower_bound || dist >= upper_bound {
                            continue;
                        }
                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                } else {
                    for (&row_id, dist) in row_ids.zip(dists) {
                        let dist = dist.into();
                        if res.len() < k {
                            res.push(OrderedNode::new(row_id, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_id, dist));
                        }
                    }
                }
            }
            false => {
                metrics.record_comparisons(storage.len());
                let row_addr_mask = prefilter.mask();
                if is_range_query {
                    let lower_bound = params.lower_bound.unwrap_or(f32::MIN).into();
                    let upper_bound = params.upper_bound.unwrap_or(f32::MAX).into();
                    for (id, &row_addr) in row_ids.enumerate() {
                        if !row_addr_mask.selected(row_addr) {
                            continue;
                        }
                        let dist = dist_calc.distance(id as u32).into();
                        if dist < lower_bound || dist >= upper_bound {
                            continue;
                        }

                        if res.len() < k {
                            res.push(OrderedNode::new(row_addr, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_addr, dist));
                        }
                    }
                } else {
                    for (id, &row_addr) in row_ids.enumerate() {
                        if !row_addr_mask.selected(row_addr) {
                            continue;
                        }

                        let dist = dist_calc.distance(id as u32).into();
                        if res.len() < k {
                            res.push(OrderedNode::new(row_addr, dist));
                        } else if res.peek().unwrap().dist > dist {
                            res.pop();
                            res.push(OrderedNode::new(row_addr, dist));
                        }
                    }
                }
            }
        };

        // we don't need to sort the results by distances here
        // because there's a SortExec node in the query plan which sorts the results from all partitions
        let (row_ids, dists): (Vec<_>, Vec<_>) = res.into_iter().map(|r| (r.id, r.dist.0)).unzip();
        let (row_ids, dists) = (UInt64Array::from(row_ids), Float32Array::from(dists));

        Ok(RecordBatch::try_new(
            ANN_SEARCH_SCHEMA.clone(),
            vec![Arc::new(dists), Arc::new(row_ids)],
        )?)
    }

    fn load(_: RecordBatch) -> Result<Self> {
        Ok(Self {})
    }

    fn index_vectors(_: &impl VectorStore, _: Self::BuildParams) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn remap(&self, _: &HashMap<u64, Option<u64>>, _: &impl VectorStore) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_batch(&self) -> Result<RecordBatch> {
        Ok(RecordBatch::new_empty(Schema::empty().into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct FlatMetadata {
    pub dim: usize,
}

#[async_trait::async_trait]
impl QuantizerMetadata for FlatMetadata {
    async fn load(_: &PreviousFileReader) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct FlatQuantizer {
    dim: usize,
    distance_type: DistanceType,
}

impl FlatQuantizer {
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        Self { dim, distance_type }
    }
}

impl Quantization for FlatQuantizer {
    type BuildParams = ();
    type Metadata = FlatMetadata;
    type Storage = FlatFloatStorage;

    fn build(data: &dyn Array, distance_type: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length();
        Ok(Self::new(dim as usize, distance_type))
    }

    fn retrain(&mut self, _: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        FLAT_COLUMN
    }

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Flat(Self {
            dim: metadata.dim,
            distance_type,
        }))
    }

    fn metadata(&self, _: Option<crate::vector::quantizer::QuantizationMetadata>) -> FlatMetadata {
        FlatMetadata { dim: self.dim }
    }

    fn metadata_key() -> &'static str {
        "flat"
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Flat
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        Ok(vectors.slice(0, vectors.len()))
    }

    fn field(&self) -> Field {
        Field::new(
            FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                self.dim as i32,
            ),
            true,
        )
    }
}

impl From<FlatQuantizer> for Quantizer {
    fn from(value: FlatQuantizer) -> Self {
        Self::Flat(value)
    }
}

impl TryFrom<Quantizer> for FlatQuantizer {
    type Error = Error;

    fn try_from(value: Quantizer) -> Result<Self> {
        match value {
            Quantizer::Flat(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input("quantizer is not FlatQuantizer")),
        }
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct FlatBinQuantizer {
    dim: usize,
    distance_type: DistanceType,
}

impl FlatBinQuantizer {
    pub fn new(dim: usize, distance_type: DistanceType) -> Self {
        Self { dim, distance_type }
    }
}

impl Quantization for FlatBinQuantizer {
    type BuildParams = ();
    type Metadata = FlatMetadata;
    type Storage = FlatBinStorage;

    fn build(data: &dyn Array, distance_type: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length();
        Ok(Self::new(dim as usize, distance_type))
    }

    fn retrain(&mut self, _: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        FLAT_COLUMN
    }

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::FlatBin(Self {
            dim: metadata.dim,
            distance_type,
        }))
    }

    fn metadata(&self, _: Option<crate::vector::quantizer::QuantizationMetadata>) -> FlatMetadata {
        FlatMetadata { dim: self.dim }
    }

    fn metadata_key() -> &'static str {
        "flat"
    }

    fn quantization_type() -> QuantizationType {
        QuantizationType::Flat
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        Ok(vectors.slice(0, vectors.len()))
    }

    fn field(&self) -> Field {
        Field::new(
            FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.dim as i32,
            ),
            true,
        )
    }
}

impl From<FlatBinQuantizer> for Quantizer {
    fn from(value: FlatBinQuantizer) -> Self {
        Self::FlatBin(value)
    }
}

impl TryFrom<Quantizer> for FlatBinQuantizer {
    type Error = Error;

    fn try_from(value: Quantizer) -> Result<Self> {
        match value {
            Quantizer::FlatBin(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input("quantizer is not FlatBinQuantizer")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
    use async_trait::async_trait;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::mask::RowAddrMask;
    use lance_core::{ROW_ID, Result};
    use lance_linalg::distance::DistanceType;

    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::prefilter::{NoFilter, PreFilter};
    use crate::vector::bq::storage::{RABIT_CODE_COLUMN, RabitQuantizationStorage};
    use crate::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
    use crate::vector::bq::{RQRotationType, builder::RabitQuantizer};
    use crate::vector::quantizer::{Quantization, QuantizerStorage};
    use crate::vector::storage::{DistCalculator, VectorStore};
    use crate::vector::v3::subindex::IvfSubIndex;

    use super::{FlatIndex, FlatQueryParams};

    struct PassAllFilter;

    #[async_trait]
    impl PreFilter for PassAllFilter {
        async fn wait_for_ready(&self) -> Result<()> {
            Ok(())
        }

        fn is_empty(&self) -> bool {
            false
        }

        fn mask(&self) -> Arc<RowAddrMask> {
            Arc::new(RowAddrMask::all_rows())
        }

        fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
            row_ids.enumerate().map(|(idx, _)| idx as u64).collect()
        }
    }

    fn make_rq_storage(num_rows: usize) -> (RabitQuantizationStorage, ArrayRef) {
        let code_dim = 64;
        let quantizer = RabitQuantizer::new_with_rotation::<arrow::datatypes::Float32Type>(
            1,
            code_dim,
            RQRotationType::Fast,
        );
        let values = Float32Array::from_iter_values(
            (0..num_rows * code_dim as usize).map(|idx| idx as f32 / 17.0),
        );
        let vectors = FixedSizeListArray::try_new_from_values(values, code_dim).unwrap();
        let codes = quantizer
            .quantize(&vectors)
            .unwrap()
            .as_fixed_size_list()
            .clone();
        let batch = RecordBatch::try_from_iter(vec![
            (
                ROW_ID,
                Arc::new(UInt64Array::from_iter_values(0..num_rows as u64)) as ArrayRef,
            ),
            (RABIT_CODE_COLUMN, Arc::new(codes) as ArrayRef),
            (
                ADD_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|idx| idx as f32 * 0.01),
                )) as ArrayRef,
            ),
            (
                SCALE_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|idx| -(1.0 + idx as f32 * 0.01)),
                )) as ArrayRef,
            ),
        ])
        .unwrap();
        let storage = RabitQuantizationStorage::try_from_batch(
            batch,
            &quantizer.metadata(None),
            DistanceType::L2,
            None,
        )
        .unwrap();
        let query: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..code_dim).map(|idx| idx as f32 / 23.0),
        ));
        (storage, query)
    }

    fn sort_result_batch(batch: &RecordBatch) -> Vec<(u64, f32)> {
        let row_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values();
        let distances = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values();
        let mut pairs = row_ids
            .iter()
            .copied()
            .zip(distances.iter().copied())
            .collect::<Vec<_>>();
        pairs.sort_by(|(lhs_id, lhs_dist), (rhs_id, rhs_dist)| {
            lhs_dist
                .total_cmp(rhs_dist)
                .then_with(|| lhs_id.cmp(rhs_id))
        });
        pairs
    }

    #[test]
    fn test_rq_flat_search_matches_full_scan_without_filter() {
        let (storage, query) = make_rq_storage(96);
        let index = FlatIndex::default();
        let metrics = LocalMetricsCollector::default();
        let params = FlatQueryParams::default();

        let result = index
            .search(
                query.clone(),
                10,
                params,
                &storage,
                Arc::new(NoFilter),
                &metrics,
            )
            .unwrap();

        let baseline = {
            let dist_calc = storage.dist_calculator(query, 0.0);
            let mut pairs = storage
                .row_ids_slice()
                .iter()
                .copied()
                .zip(dist_calc.distance_all(10))
                .collect::<Vec<_>>();
            pairs.sort_by(|(lhs_id, lhs_dist), (rhs_id, rhs_dist)| {
                lhs_dist
                    .total_cmp(rhs_dist)
                    .then_with(|| lhs_id.cmp(rhs_id))
            });
            pairs.truncate(10);
            pairs
        };

        assert_eq!(sort_result_batch(&result), baseline);
        assert_eq!(
            metrics.comparisons.load(Ordering::Relaxed)
                + metrics.pruned_rows.load(Ordering::Relaxed),
            storage.len()
        );
    }

    #[test]
    fn test_rq_flat_range_query_matches_scalar_path() {
        let (storage, query) = make_rq_storage(64);
        let index = FlatIndex::default();
        let metrics = NoOpMetricsCollector;
        let params = FlatQueryParams {
            lower_bound: Some(-1000.0),
            upper_bound: Some(1000.0),
            dist_q_c: 0.0,
        };

        let result = index
            .search(
                query.clone(),
                8,
                params,
                &storage,
                Arc::new(NoFilter),
                &metrics,
            )
            .unwrap();

        let dist_calc = storage.dist_calculator(query, 0.0);
        let mut baseline = storage
            .row_ids_slice()
            .iter()
            .copied()
            .zip(dist_calc.distance_all(8))
            .collect::<Vec<_>>();
        baseline.sort_by(|(lhs_id, lhs_dist), (rhs_id, rhs_dist)| {
            lhs_dist
                .total_cmp(rhs_dist)
                .then_with(|| lhs_id.cmp(rhs_id))
        });
        baseline.truncate(8);

        assert_eq!(sort_result_batch(&result), baseline);
    }

    #[test]
    fn test_rq_flat_prefilter_matches_scalar_path() {
        let (storage, query) = make_rq_storage(64);
        let index = FlatIndex::default();
        let metrics = NoOpMetricsCollector;

        let result = index
            .search(
                query.clone(),
                8,
                FlatQueryParams::default(),
                &storage,
                Arc::new(PassAllFilter),
                &metrics,
            )
            .unwrap();

        let dist_calc = storage.dist_calculator(query, 0.0);
        let mut baseline = storage
            .row_ids()
            .enumerate()
            .map(|(idx, row_id)| (*row_id, dist_calc.distance(idx as u32)))
            .collect::<Vec<_>>();
        baseline.sort_by(|(lhs_id, lhs_dist), (rhs_id, rhs_dist)| {
            lhs_dist
                .total_cmp(rhs_dist)
                .then_with(|| lhs_id.cmp(rhs_id))
        });
        baseline.truncate(8);

        assert_eq!(sort_result_batch(&result), baseline);
    }
}
