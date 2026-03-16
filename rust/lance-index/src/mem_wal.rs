// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Error;
use lance_table::format::pb;
use murmur3::murmur3_32;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Index, IndexType};

pub const MEM_WAL_INDEX_NAME: &str = "__lance_mem_wal";

/// Type alias for region identifier (UUID v4).
pub type RegionId = Uuid;

/// A flushed MemTable generation and its storage location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FlushedGeneration {
    pub generation: u64,
    pub path: String,
}

impl From<&FlushedGeneration> for pb::FlushedGeneration {
    fn from(fg: &FlushedGeneration) -> Self {
        Self {
            generation: fg.generation,
            path: fg.path.clone(),
        }
    }
}

impl From<pb::FlushedGeneration> for FlushedGeneration {
    fn from(fg: pb::FlushedGeneration) -> Self {
        Self {
            generation: fg.generation,
            path: fg.path,
        }
    }
}

/// A region's merged generation, used in MemWalIndexDetails.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MergedGeneration {
    pub region_id: Uuid,
    pub generation: u64,
}

impl DeepSizeOf for MergedGeneration {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0 // UUID is 16 bytes fixed size, no heap allocations
    }
}

impl MergedGeneration {
    pub fn new(region_id: Uuid, generation: u64) -> Self {
        Self {
            region_id,
            generation,
        }
    }
}

impl From<&MergedGeneration> for pb::MergedGeneration {
    fn from(mg: &MergedGeneration) -> Self {
        Self {
            region_id: Some((&mg.region_id).into()),
            generation: mg.generation,
        }
    }
}

impl TryFrom<pb::MergedGeneration> for MergedGeneration {
    type Error = Error;

    fn try_from(mg: pb::MergedGeneration) -> lance_core::Result<Self> {
        let region_id = mg
            .region_id
            .as_ref()
            .map(Uuid::try_from)
            .ok_or_else(|| Error::invalid_input("Missing region_id in MergedGeneration"))??;
        Ok(Self {
            region_id,
            generation: mg.generation,
        })
    }
}

/// Tracks which merged generation a base table index has been rebuilt to cover.
/// Used to determine whether to read from flushed MemTable indexes or base table.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct IndexCatchupProgress {
    pub index_name: String,
    pub caught_up_generations: Vec<MergedGeneration>,
}

impl IndexCatchupProgress {
    pub fn new(index_name: String, caught_up_generations: Vec<MergedGeneration>) -> Self {
        Self {
            index_name,
            caught_up_generations,
        }
    }

    /// Get the caught up generation for a specific region.
    /// Returns None if the region is not present (assumed fully caught up).
    pub fn caught_up_generation_for_region(&self, region_id: &Uuid) -> Option<u64> {
        self.caught_up_generations
            .iter()
            .find(|mg| &mg.region_id == region_id)
            .map(|mg| mg.generation)
    }
}

impl From<&IndexCatchupProgress> for pb::IndexCatchupProgress {
    fn from(icp: &IndexCatchupProgress) -> Self {
        Self {
            index_name: icp.index_name.clone(),
            caught_up_generations: icp
                .caught_up_generations
                .iter()
                .map(|mg| mg.into())
                .collect(),
        }
    }
}

impl TryFrom<pb::IndexCatchupProgress> for IndexCatchupProgress {
    type Error = Error;

    fn try_from(icp: pb::IndexCatchupProgress) -> lance_core::Result<Self> {
        Ok(Self {
            index_name: icp.index_name,
            caught_up_generations: icp
                .caught_up_generations
                .into_iter()
                .map(MergedGeneration::try_from)
                .collect::<lance_core::Result<_>>()?,
        })
    }
}

/// Region manifest containing epoch-based fencing and WAL state.
/// Each region has exactly one active writer at any time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionManifest {
    pub region_id: Uuid,
    pub version: u64,
    pub region_spec_id: u32,
    pub writer_epoch: u64,
    /// The most recent WAL entry position (0-based) flushed to a MemTable.
    /// Recovery replays from `replay_after_wal_entry_position + 1`.
    pub replay_after_wal_entry_position: u64,
    /// The most recent WAL entry position (0-based) when manifest was updated.
    pub wal_entry_position_last_seen: u64,
    pub current_generation: u64,
    pub flushed_generations: Vec<FlushedGeneration>,
}

impl DeepSizeOf for RegionManifest {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.flushed_generations.deep_size_of_children(context)
    }
}

impl From<&RegionManifest> for pb::RegionManifest {
    fn from(rm: &RegionManifest) -> Self {
        Self {
            region_id: Some((&rm.region_id).into()),
            version: rm.version,
            region_spec_id: rm.region_spec_id,
            writer_epoch: rm.writer_epoch,
            replay_after_wal_entry_position: rm.replay_after_wal_entry_position,
            wal_entry_position_last_seen: rm.wal_entry_position_last_seen,
            current_generation: rm.current_generation,
            flushed_generations: rm.flushed_generations.iter().map(|fg| fg.into()).collect(),
        }
    }
}

impl TryFrom<pb::RegionManifest> for RegionManifest {
    type Error = Error;

    fn try_from(rm: pb::RegionManifest) -> lance_core::Result<Self> {
        let region_id = rm
            .region_id
            .as_ref()
            .map(Uuid::try_from)
            .ok_or_else(|| Error::invalid_input("Missing region_id in RegionManifest"))??;
        Ok(Self {
            region_id,
            version: rm.version,
            region_spec_id: rm.region_spec_id,
            writer_epoch: rm.writer_epoch,
            replay_after_wal_entry_position: rm.replay_after_wal_entry_position,
            wal_entry_position_last_seen: rm.wal_entry_position_last_seen,
            current_generation: rm.current_generation,
            flushed_generations: rm
                .flushed_generations
                .into_iter()
                .map(FlushedGeneration::from)
                .collect(),
        })
    }
}

/// Region field definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RegionField {
    pub field_id: String,
    pub source_ids: Vec<i32>,
    pub transform: Option<String>,
    pub expression: Option<String>,
    pub result_type: String,
    pub parameters: HashMap<String, String>,
}

impl From<&RegionField> for pb::RegionField {
    fn from(rf: &RegionField) -> Self {
        Self {
            field_id: rf.field_id.clone(),
            source_ids: rf.source_ids.clone(),
            transform: rf.transform.clone(),
            expression: rf.expression.clone(),
            result_type: rf.result_type.clone(),
            parameters: rf.parameters.clone(),
        }
    }
}

impl From<pb::RegionField> for RegionField {
    fn from(rf: pb::RegionField) -> Self {
        Self {
            field_id: rf.field_id,
            source_ids: rf.source_ids,
            transform: rf.transform,
            expression: rf.expression,
            result_type: rf.result_type,
            parameters: rf.parameters,
        }
    }
}

/// Region spec definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RegionSpec {
    pub spec_id: u32,
    pub fields: Vec<RegionField>,
}

impl From<&RegionSpec> for pb::RegionSpec {
    fn from(rs: &RegionSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.iter().map(|f| f.into()).collect(),
        }
    }
}

impl From<pb::RegionSpec> for RegionSpec {
    fn from(rs: pb::RegionSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.into_iter().map(RegionField::from).collect(),
        }
    }
}

/// Well-known transform types for region bucket transforms.
pub mod transforms {
    /// Identity transform - uses the value as-is.
    pub const IDENTITY: &str = "identity";
    /// Bucket transform - hashes the value into N buckets.
    pub const BUCKET: &str = "bucket";
    /// Multi-bucket transform - same as bucket but multiple fields.
    pub const MULTI_BUCKET: &str = "multi_bucket";
}

/// Bucket transform result containing the bucket number and region values.
#[derive(Debug, Clone)]
pub struct BucketResult {
    /// The computed bucket number (0 to num_buckets-1).
    pub bucket: u32,
    /// The number of buckets in the transform.
    pub num_buckets: u32,
}

impl RegionField {
    /// Compute the value of this region field for a single row.
    /// Returns the raw bytes representing the hashed or transformed value.
    pub fn compute_value(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> lance_core::Result<Vec<u8>> {
        match self.transform.as_deref() {
            Some(transforms::BUCKET) | Some(transforms::MULTI_BUCKET) => {
                self.compute_bucket_value(batch, row_idx)
            }
            Some(transforms::IDENTITY) | None => self.compute_identity_value(batch, row_idx),
            Some(transform) => Err(Error::not_supported_source(
                format!("Unsupported region transform: {}", transform).into(),
            )),
        }
    }

    /// Compute the bucket value for bucket/multi_bucket transforms.
    fn compute_bucket_value(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> lance_core::Result<Vec<u8>> {
        let num_buckets: u32 = self
            .parameters
            .get("num_buckets")
            .ok_or_else(|| Error::invalid_input("bucket transform requires num_buckets"))?
            .parse()
            .map_err(|e| Error::invalid_input(format!("Invalid num_buckets: {}", e)))?;

        let hash = self.compute_murmur3_hash(batch, row_idx)?;
        let bucket = (hash.unsigned_abs()) % num_buckets;
        Ok(bucket.to_le_bytes().to_vec())
    }

    /// Compute the identity value (raw column values concatenated).
    fn compute_identity_value(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> lance_core::Result<Vec<u8>> {
        let mut data = Vec::new();
        for &field_id in &self.source_ids {
            let col_idx = find_column_by_field_id(batch, field_id)?;
            let col = batch.column(col_idx);
            append_array_value(&mut data, col.as_ref(), row_idx)?;
        }
        Ok(data)
    }

    /// Compute the murmur3 hash for the source columns at a specific row.
    fn compute_murmur3_hash(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> lance_core::Result<i32> {
        let mut data = Vec::new();
        for &field_id in &self.source_ids {
            let col_idx = find_column_by_field_id(batch, field_id)?;
            let col = batch.column(col_idx);
            append_array_value(&mut data, col.as_ref(), row_idx)?;
        }

        let hash = murmur3_32(&mut Cursor::new(&data), 0)
            .map_err(|e| Error::internal(format!("murmur3 hash failed: {}", e)))?;
        Ok(hash as i32)
    }
}

impl RegionSpec {
    /// Compute the bucket for a given row based on the region spec fields.
    ///
    /// This method assumes the first field in the spec defines the bucket transform.
    /// For multi-field bucket specs, all field hashes are combined.
    ///
    /// # Arguments
    ///
    /// * `batch` - The RecordBatch containing the row data
    /// * `row_idx` - The index of the row to compute the bucket for
    ///
    /// # Returns
    ///
    /// A `BucketResult` containing the bucket number and total bucket count.
    pub fn compute_bucket(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> lance_core::Result<BucketResult> {
        let field = self
            .fields
            .first()
            .ok_or_else(|| Error::invalid_input("Region spec has no fields"))?;

        match field.transform.as_deref() {
            Some(transforms::BUCKET) | Some(transforms::MULTI_BUCKET) => {
                let num_buckets: u32 = field
                    .parameters
                    .get("num_buckets")
                    .ok_or_else(|| Error::invalid_input("bucket transform requires num_buckets"))?
                    .parse()
                    .map_err(|e| Error::invalid_input(format!("Invalid num_buckets: {}", e)))?;

                let hash = field.compute_murmur3_hash(batch, row_idx)?;
                let bucket = (hash.unsigned_abs()) % num_buckets;

                Ok(BucketResult {
                    bucket,
                    num_buckets,
                })
            }
            _ => Err(Error::not_supported_source(
                "Only bucket/multi_bucket transforms are currently supported for routing".into(),
            )),
        }
    }

    /// Compute buckets for all rows in a batch.
    ///
    /// Returns a vector of bucket assignments, one per row.
    pub fn compute_buckets(
        &self,
        batch: &RecordBatch,
    ) -> lance_core::Result<(Vec<u32>, u32)> {
        let field = self
            .fields
            .first()
            .ok_or_else(|| Error::invalid_input("Region spec has no fields"))?;

        let num_buckets: u32 = field
            .parameters
            .get("num_buckets")
            .ok_or_else(|| Error::invalid_input("bucket transform requires num_buckets"))?
            .parse()
            .map_err(|e| Error::invalid_input(format!("Invalid num_buckets: {}", e)))?;

        let buckets: Vec<u32> = (0..batch.num_rows())
            .map(|row_idx| {
                let hash = field.compute_murmur3_hash(batch, row_idx)?;
                Ok((hash.unsigned_abs()) % num_buckets)
            })
            .collect::<lance_core::Result<_>>()?;

        Ok((buckets, num_buckets))
    }
}

/// Find the column index in a RecordBatch by field ID.
///
/// Field IDs are stored in schema metadata. If not found, falls back to index-based lookup.
fn find_column_by_field_id(batch: &RecordBatch, field_id: i32) -> lance_core::Result<usize> {
    // First, try to find by field ID in schema metadata
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        if let Some(id_str) = field.metadata().get("field_id") {
            if let Ok(id) = id_str.parse::<i32>() {
                if id == field_id {
                    return Ok(idx);
                }
            }
        }
    }

    // Fallback: treat field_id as column index
    let idx = field_id as usize;
    if idx < batch.num_columns() {
        Ok(idx)
    } else {
        Err(Error::invalid_input(format!(
            "Field ID {} not found in batch with {} columns",
            field_id,
            batch.num_columns()
        )))
    }
}

/// Append the value of an array at a specific row index to a byte buffer.
fn append_array_value(
    buffer: &mut Vec<u8>,
    array: &dyn Array,
    row_idx: usize,
) -> lance_core::Result<()> {
    use arrow_array::{
        BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int8Array, LargeBinaryArray, LargeStringArray, StringArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };

    if array.is_null(row_idx) {
        // For null values, append a null marker
        buffer.push(0);
        return Ok(());
    }

    // Non-null marker
    buffer.push(1);

    // Type-specific value extraction
    // We use a simple approach: convert to bytes and append
    if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        buffer.extend_from_slice(&arr.value(row_idx).to_le_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
        buffer.push(if arr.value(row_idx) { 1 } else { 0 });
    } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        let val = arr.value(row_idx);
        buffer.extend_from_slice(&(val.len() as u32).to_le_bytes());
        buffer.extend_from_slice(val.as_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
        let val = arr.value(row_idx);
        buffer.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buffer.extend_from_slice(val.as_bytes());
    } else if let Some(arr) = array.as_any().downcast_ref::<BinaryArray>() {
        let val = arr.value(row_idx);
        buffer.extend_from_slice(&(val.len() as u32).to_le_bytes());
        buffer.extend_from_slice(val);
    } else if let Some(arr) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        let val = arr.value(row_idx);
        buffer.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buffer.extend_from_slice(val);
    } else {
        return Err(Error::not_supported_source(
            format!(
                "Unsupported array type for bucket hashing: {:?}",
                array.data_type()
            )
            .into(),
        ));
    }

    Ok(())
}

/// Index details for MemWAL Index, stored in IndexMetadata.index_details.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWalIndexDetails {
    pub snapshot_ts_millis: i64,
    pub num_regions: u32,
    pub inline_snapshots: Option<Vec<u8>>,
    pub region_specs: Vec<RegionSpec>,
    pub maintained_indexes: Vec<String>,
    pub merged_generations: Vec<MergedGeneration>,
    pub index_catchup: Vec<IndexCatchupProgress>,
}

impl From<&MemWalIndexDetails> for pb::MemWalIndexDetails {
    fn from(details: &MemWalIndexDetails) -> Self {
        Self {
            snapshot_ts_millis: details.snapshot_ts_millis,
            num_regions: details.num_regions,
            inline_snapshots: details.inline_snapshots.clone(),
            region_specs: details.region_specs.iter().map(|rs| rs.into()).collect(),
            maintained_indexes: details.maintained_indexes.clone(),
            merged_generations: details
                .merged_generations
                .iter()
                .map(|mg| mg.into())
                .collect(),
            index_catchup: details.index_catchup.iter().map(|icp| icp.into()).collect(),
        }
    }
}

impl TryFrom<pb::MemWalIndexDetails> for MemWalIndexDetails {
    type Error = Error;

    fn try_from(details: pb::MemWalIndexDetails) -> lance_core::Result<Self> {
        Ok(Self {
            snapshot_ts_millis: details.snapshot_ts_millis,
            num_regions: details.num_regions,
            inline_snapshots: details.inline_snapshots,
            region_specs: details
                .region_specs
                .into_iter()
                .map(RegionSpec::from)
                .collect(),
            maintained_indexes: details.maintained_indexes,
            merged_generations: details
                .merged_generations
                .into_iter()
                .map(MergedGeneration::try_from)
                .collect::<lance_core::Result<_>>()?,
            index_catchup: details
                .index_catchup
                .into_iter()
                .map(IndexCatchupProgress::try_from)
                .collect::<lance_core::Result<_>>()?,
        })
    }
}

/// MemWAL Index provides access to MemWAL configuration and state.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct MemWalIndex {
    pub details: MemWalIndexDetails,
}

impl MemWalIndex {
    pub fn new(details: MemWalIndexDetails) -> Self {
        Self { details }
    }

    pub fn merged_generation_for_region(&self, region_id: &Uuid) -> Option<u64> {
        self.details
            .merged_generations
            .iter()
            .find(|mg| &mg.region_id == region_id)
            .map(|mg| mg.generation)
    }

    /// Get the caught up generation for a specific index and region.
    /// Returns None if the index is not tracked (assumed fully caught up).
    pub fn index_caught_up_generation(&self, index_name: &str, region_id: &Uuid) -> Option<u64> {
        self.details
            .index_catchup
            .iter()
            .find(|icp| icp.index_name == index_name)
            .and_then(|icp| icp.caught_up_generation_for_region(region_id))
    }

    /// Check if an index is fully caught up for a region.
    /// Returns true if the index covers all merged data for the region.
    pub fn is_index_caught_up(&self, index_name: &str, region_id: &Uuid) -> bool {
        let merged_gen = self.merged_generation_for_region(region_id).unwrap_or(0);
        let caught_up_gen = self.index_caught_up_generation(index_name, region_id);

        // If not tracked in index_catchup, assumed fully caught up
        caught_up_gen.is_none_or(|generation| generation >= merged_gen)
    }
}

#[derive(Serialize)]
struct MemWalStatistics {
    num_regions: u32,
    num_merged_generations: usize,
    num_region_specs: usize,
    num_maintained_indexes: usize,
    num_index_catchup_entries: usize,
}

#[async_trait]
impl Index for MemWalIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> lance_core::Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::not_supported_source(
            "MemWalIndex is not a vector index".into(),
        ))
    }

    fn statistics(&self) -> lance_core::Result<serde_json::Value> {
        let stats = MemWalStatistics {
            num_regions: self.details.num_regions,
            num_merged_generations: self.details.merged_generations.len(),
            num_region_specs: self.details.region_specs.len(),
            num_maintained_indexes: self.details.maintained_indexes.len(),
            num_index_catchup_entries: self.details.index_catchup.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
                "failed to serialize MemWAL index statistics: {}",
                e
            ))
        })
    }

    async fn prewarm(&self) -> lance_core::Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::MemWal
    }

    async fn calculate_included_frags(&self) -> lance_core::Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let id_array = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let name_array = StringArray::from(vec![
            "alice", "bob", "charlie", "david", "eve", "frank", "grace", "henry", "ivy", "jack",
        ]);
        let value_array = Int32Array::from(vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(value_array),
            ],
        )
        .unwrap()
    }

    fn make_bucket_region_spec(source_id: i32, num_buckets: u32) -> RegionSpec {
        let mut parameters = HashMap::new();
        parameters.insert("num_buckets".to_string(), num_buckets.to_string());

        RegionSpec {
            spec_id: 1,
            fields: vec![RegionField {
                field_id: "bucket_field".to_string(),
                source_ids: vec![source_id],
                transform: Some(transforms::BUCKET.to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters,
            }],
        }
    }

    #[test]
    fn test_compute_bucket_single_field() {
        let batch = make_test_batch();
        let spec = make_bucket_region_spec(0, 4); // Use column 0 (id), 4 buckets

        for row_idx in 0..batch.num_rows() {
            let result = spec.compute_bucket(&batch, row_idx).unwrap();
            assert!(result.bucket < 4, "Bucket should be in range [0, 4)");
            assert_eq!(result.num_buckets, 4);
        }
    }

    #[test]
    fn test_compute_buckets_batch() {
        let batch = make_test_batch();
        let spec = make_bucket_region_spec(0, 4);

        let (buckets, num_buckets) = spec.compute_buckets(&batch).unwrap();

        assert_eq!(buckets.len(), batch.num_rows());
        assert_eq!(num_buckets, 4);

        for &bucket in &buckets {
            assert!(bucket < 4, "All buckets should be in range [0, 4)");
        }
    }

    #[test]
    fn test_compute_bucket_string_field() {
        let batch = make_test_batch();
        let spec = make_bucket_region_spec(1, 8); // Use column 1 (name), 8 buckets

        let (buckets, num_buckets) = spec.compute_buckets(&batch).unwrap();

        assert_eq!(buckets.len(), batch.num_rows());
        assert_eq!(num_buckets, 8);

        for &bucket in &buckets {
            assert!(bucket < 8, "All buckets should be in range [0, 8)");
        }
    }

    #[test]
    fn test_compute_bucket_deterministic() {
        let batch = make_test_batch();
        let spec = make_bucket_region_spec(0, 16);

        let (buckets1, _) = spec.compute_buckets(&batch).unwrap();
        let (buckets2, _) = spec.compute_buckets(&batch).unwrap();

        assert_eq!(buckets1, buckets2, "Bucket assignments should be deterministic");
    }

    #[test]
    fn test_unsupported_transform() {
        let spec = RegionSpec {
            spec_id: 1,
            fields: vec![RegionField {
                field_id: "unknown_field".to_string(),
                source_ids: vec![0],
                transform: Some("year".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::new(),
            }],
        };

        let batch = make_test_batch();
        let result = spec.compute_bucket(&batch, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_num_buckets() {
        let spec = RegionSpec {
            spec_id: 1,
            fields: vec![RegionField {
                field_id: "bucket_field".to_string(),
                source_ids: vec![0],
                transform: Some(transforms::BUCKET.to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::new(), // Missing num_buckets
            }],
        };

        let batch = make_test_batch();
        let result = spec.compute_bucket(&batch, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_multi_column_bucket() {
        let mut parameters = HashMap::new();
        parameters.insert("num_buckets".to_string(), "4".to_string());

        let spec = RegionSpec {
            spec_id: 1,
            fields: vec![RegionField {
                field_id: "multi_bucket".to_string(),
                source_ids: vec![0, 2], // id and value columns
                transform: Some(transforms::MULTI_BUCKET.to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters,
            }],
        };

        let batch = make_test_batch();
        let (buckets, num_buckets) = spec.compute_buckets(&batch).unwrap();

        assert_eq!(buckets.len(), batch.num_rows());
        assert_eq!(num_buckets, 4);

        for &bucket in &buckets {
            assert!(bucket < 4);
        }
    }
}
