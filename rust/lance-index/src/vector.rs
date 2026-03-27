// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index
//!

use std::any::Any;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt32Array};
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use ivf::storage::IvfModel;
use lance_core::cache::{CacheCodec, CountingWriter, read_type_tag};
use lance_core::{ROW_ID_FIELD, Result};
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use prost::Message;
use quantizer::{QuantizationType, Quantizer};
use std::sync::LazyLock;
use v3::subindex::SubIndexType;

pub mod bq;
pub mod distributed;
pub mod flat;
pub mod graph;
pub mod hnsw;
pub mod ivf;
pub mod kmeans;
pub mod pq;
pub mod quantizer;
pub mod residual;
pub mod shared;
pub mod sq;
pub mod storage;
pub mod transform;
pub mod utils;
pub mod v3;

use super::pb;
use crate::metrics::MetricsCollector;
use crate::{Index, prefilter::PreFilter};

// TODO: Make these crate private once the migration from lance to lance-index is done.
pub const DIST_COL: &str = "_distance";
pub const DISTANCE_TYPE_KEY: &str = "distance_type";
pub const INDEX_UUID_COLUMN: &str = "__index_uuid";
pub const PART_ID_COLUMN: &str = "__ivf_part_id";
pub const DIST_Q_C_COLUMN: &str = "__dist_q_c";
// dist from vector to centroid
pub const CENTROID_DIST_COLUMN: &str = "__centroid_dist";
pub const PQ_CODE_COLUMN: &str = "__pq_code";
pub const SQ_CODE_COLUMN: &str = "__sq_code";
pub const LOSS_METADATA_KEY: &str = "_loss";

pub static VECTOR_RESULT_SCHEMA: LazyLock<arrow_schema::SchemaRef> = LazyLock::new(|| {
    arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![
        Field::new(DIST_COL, arrow_schema::DataType::Float32, false),
        ROW_ID_FIELD.clone(),
    ]))
});

pub static PART_ID_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(PART_ID_COLUMN, arrow_schema::DataType::UInt32, true)
});

pub static CENTROID_DIST_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(CENTROID_DIST_COLUMN, arrow_schema::DataType::Float32, true)
});

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    /// The column to be searched.
    pub column: String,

    /// The vector to be searched.
    pub key: ArrayRef,

    /// Top k results to return.
    pub k: usize,

    /// The lower bound (inclusive) of the distance to be searched.
    pub lower_bound: Option<f32>,

    /// The upper bound (exclusive) of the distance to be searched.
    pub upper_bound: Option<f32>,

    /// The minimum number of probes to load and search.  More partitions
    /// will only be loaded if we have not found k results, or the algorithm
    /// determines more partitions are needed to satisfy recall requirements.
    ///
    /// The planner will always search at least this many partitions. Defaults to 1.
    pub minimum_nprobes: usize,

    /// The maximum number of probes to load and search.  If not set then
    /// ALL partitions will be searched, if needed, to satisfy k results.
    pub maximum_nprobes: Option<usize>,

    /// The number of candidates to reserve while searching.
    /// this is an optional parameter for HNSW related index types.
    pub ef: Option<usize>,

    /// If presented, apply a refine step.
    /// TODO: should we support fraction / float number here?
    pub refine_factor: Option<u32>,

    /// Distance metric type. If None, uses the index's metric (if available)
    /// or the default for the data type.
    pub metric_type: Option<DistanceType>,

    /// Whether to use an ANN index if available
    pub use_index: bool,

    /// the distance between the query and the centroid
    /// this is only used for IVF index with Rabit quantization
    pub dist_q_c: f32,
}

impl From<pb::VectorMetricType> for DistanceType {
    fn from(proto: pb::VectorMetricType) -> Self {
        match proto {
            pb::VectorMetricType::L2 => Self::L2,
            pb::VectorMetricType::Cosine => Self::Cosine,
            pb::VectorMetricType::Dot => Self::Dot,
            pb::VectorMetricType::Hamming => Self::Hamming,
        }
    }
}

impl From<DistanceType> for pb::VectorMetricType {
    fn from(mt: DistanceType) -> Self {
        match mt {
            DistanceType::L2 => Self::L2,
            DistanceType::Cosine => Self::Cosine,
            DistanceType::Dot => Self::Dot,
            DistanceType::Hamming => Self::Hamming,
        }
    }
}

/// Serializable snapshot of a vector index, suitable for caching.
///
/// Implementations must be cheaply reconstructable into a live
/// [`VectorIndex`] given an ObjectStore, file metadata cache, and partition
/// cache. The reconstruction cost should be dominated by re-opening
/// `FileReader`s, which is cheap when the file metadata cache is warm.
pub trait VectorIndexData: DeepSizeOf + std::fmt::Debug + Send + Sync {
    /// Downcast to `&dyn Any` for concrete type access during reconstruction.
    fn as_any(&self) -> &dyn Any;
}

/// Deserialize a [`VectorIndexData`] from a stream previously written by
/// [`lance_core::cache::serialize_tagged`].
///
/// Reads the type tag and dispatches to the correct concrete deserializer.
pub fn deserialize_vector_index_data(reader: &mut dyn Read) -> Result<Arc<dyn VectorIndexData>> {
    let tag = read_type_tag(reader)?;
    match tag.as_str() {
        "IVF" => {
            let state = IvfIndexState::deserialize(reader)?;
            Ok(Arc::new(state))
        }
        other => Err(lance_core::Error::io(format!(
            "unknown VectorIndexData type tag: {other:?}"
        ))),
    }
}

/// Serializable state of an IVF index, sufficient to reconstruct the index
/// without re-reading global buffers from object storage.
///
/// Produced by [`VectorIndex::cacheable_state`] and consumed by a
/// reconstruction function that re-opens FileReaders using cached file metadata.
#[derive(Debug, Clone)]
pub struct IvfIndexState {
    /// Object-store path to the index file (before `to_local_path` conversion).
    pub index_file_path: String,
    pub uuid: String,
    /// IvfModel for the index file (sub-index row layout).
    pub ivf: IvfModel,
    /// IvfModel for the auxiliary/storage file (quantizer row layout).
    /// The index and aux files have independent row layouts, so we must store
    /// both to avoid using wrong row offsets during reconstruction.
    pub aux_ivf: IvfModel,
    pub distance_type: DistanceType,
    pub sub_index_metadata: Vec<String>,
    /// JSON serialization of `Q::Metadata` (quantizer-specific metadata).
    pub quantizer_metadata_json: String,
    /// Large quantizer data (PQ codebook, RQ rotation matrix) from `extra_metadata()`.
    pub quantizer_extra_data: Option<Vec<u8>>,
    pub sub_index_type: SubIndexType,
    pub quantization_type: QuantizationType,
    /// The cache key prefix used by the original index's WeakLanceCache.
    /// Needed to reconnect the reconstructed index to the shared cache backend.
    pub cache_key_prefix: String,
    /// File sizes for the index and auxiliary files, used to avoid HEAD requests
    /// when reconstructing from cache.
    pub index_file_size: u64,
    pub aux_file_size: u64,
}

/// Serialization header for [`IvfIndexState`].
#[derive(serde::Serialize, serde::Deserialize)]
struct IvfIndexStateHeader {
    index_file_path: String,
    uuid: String,
    distance_type: String,
    sub_index_metadata: Vec<String>,
    sub_index_type: String,
    quantization_type: String,
    quantizer_metadata_json: String,
    #[serde(default)]
    cache_key_prefix: String,
    #[serde(default)]
    index_file_size: u64,
    #[serde(default)]
    aux_file_size: u64,
}

/// Wire format:
/// `[header_json_len: u64 LE][header JSON][ivf_pb_len: u64 LE][ivf protobuf]
///  [extra_len: u64 LE][extra bytes][aux_ivf_pb_len: u64 LE][aux_ivf protobuf]`
impl CacheCodec for IvfIndexState {
    fn serialize(&self, writer: &mut dyn Write) -> Result<usize> {
        let header = IvfIndexStateHeader {
            index_file_path: self.index_file_path.clone(),
            uuid: self.uuid.clone(),
            distance_type: self.distance_type.to_string(),
            sub_index_metadata: self.sub_index_metadata.clone(),
            sub_index_type: self.sub_index_type.to_string(),
            quantization_type: self.quantization_type.to_string(),
            quantizer_metadata_json: self.quantizer_metadata_json.clone(),
            cache_key_prefix: self.cache_key_prefix.clone(),
            index_file_size: self.index_file_size,
            aux_file_size: self.aux_file_size,
        };
        let header_json = serde_json::to_vec(&header)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState header: {e}")))?;

        let ivf_pb = pb::Ivf::try_from(&self.ivf)?;
        let ivf_bytes = ivf_pb.encode_to_vec();

        let extra = self.quantizer_extra_data.as_deref().unwrap_or(&[]);

        let aux_ivf_pb = pb::Ivf::try_from(&self.aux_ivf)?;
        let aux_ivf_bytes = aux_ivf_pb.encode_to_vec();

        let mut cw = CountingWriter::new(writer);
        use std::io::Write as _;
        cw.write_all(&(header_json.len() as u64).to_le_bytes())?;
        cw.write_all(&header_json)?;
        cw.write_all(&(ivf_bytes.len() as u64).to_le_bytes())?;
        cw.write_all(&ivf_bytes)?;
        cw.write_all(&(extra.len() as u64).to_le_bytes())?;
        cw.write_all(extra)?;
        cw.write_all(&(aux_ivf_bytes.len() as u64).to_le_bytes())?;
        cw.write_all(&aux_ivf_bytes)?;
        Ok(cw.written())
    }

    fn type_tag(&self) -> &'static str {
        "IVF"
    }

    fn deserialize(reader: &mut dyn Read) -> Result<Self> {
        fn read_u64(r: &mut dyn Read) -> Result<u64> {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)
                .map_err(|e| lance_core::Error::io(e.to_string()))?;
            Ok(u64::from_le_bytes(buf))
        }
        fn read_bytes(r: &mut dyn Read, len: usize) -> Result<Vec<u8>> {
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)
                .map_err(|e| lance_core::Error::io(e.to_string()))?;
            Ok(buf)
        }

        let header_len = read_u64(reader)? as usize;
        let header_bytes = read_bytes(reader, header_len)?;
        let header: IvfIndexStateHeader = serde_json::from_slice(&header_bytes)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState header: {e}")))?;

        let ivf_len = read_u64(reader)? as usize;
        let ivf_bytes = read_bytes(reader, ivf_len)?;
        let ivf_pb = pb::Ivf::decode(ivf_bytes.as_slice())
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState IVF decode: {e}")))?;
        let ivf = IvfModel::try_from(ivf_pb)?;

        let extra_len = read_u64(reader)? as usize;
        let quantizer_extra_data = if extra_len > 0 {
            Some(read_bytes(reader, extra_len)?)
        } else {
            None
        };

        let aux_ivf = match read_u64(reader) {
            Ok(aux_ivf_len) => {
                let aux_ivf_bytes = read_bytes(reader, aux_ivf_len as usize)?;
                let aux_ivf_pb = pb::Ivf::decode(aux_ivf_bytes.as_slice()).map_err(|e| {
                    lance_core::Error::io(format!("IvfIndexState aux IVF decode: {e}"))
                })?;
                IvfModel::try_from(aux_ivf_pb)?
            }
            // Legacy format without aux_ivf — fall back to ivf.
            Err(_) => ivf.clone(),
        };

        let distance_type = DistanceType::try_from(header.distance_type.as_str())?;
        let sub_index_type = SubIndexType::try_from(header.sub_index_type.as_str())?;
        let quantization_type = header.quantization_type.parse::<QuantizationType>()?;

        Ok(Self {
            index_file_path: header.index_file_path,
            uuid: header.uuid,
            ivf,
            aux_ivf,
            distance_type,
            sub_index_metadata: header.sub_index_metadata,
            quantizer_metadata_json: header.quantizer_metadata_json,
            quantizer_extra_data,
            sub_index_type,
            quantization_type,
            cache_key_prefix: header.cache_key_prefix,
            index_file_size: header.index_file_size,
            aux_file_size: header.aux_file_size,
        })
    }
}

impl DeepSizeOf for IvfIndexState {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.index_file_path.deep_size_of_children(context)
            + self.uuid.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.aux_ivf.deep_size_of_children(context)
            + self.sub_index_metadata.deep_size_of_children(context)
            + self.quantizer_metadata_json.deep_size_of_children(context)
            + self
                .quantizer_extra_data
                .as_ref()
                .map(|v| v.deep_size_of_children(context))
                .unwrap_or(0)
            + self.cache_key_prefix.deep_size_of_children(context)
    }
}

impl VectorIndexData for IvfIndexState {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
///
/// Vector indices are often built as a chain of indices.  For example, IVF -> PQ
/// or IVF -> HNSW -> SQ.
///
/// We use one trait for both the top-level and the sub-indices.  Typically the top-level
/// search is a partition-aware search and all sub-indices are whole-index searches.
#[async_trait]
#[allow(clippy::redundant_pub_crate)]
pub trait VectorIndex: Send + Sync + std::fmt::Debug + Index {
    /// Search entire index for k nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, true),
    ///   Field::new("_distance", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// The `pre_filter` argument is used to filter out row ids that we know are
    /// not relevant to the query. For example, it removes deleted rows or rows that
    /// do not match a user-provided filter.
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// Find partitions that may contain nearest neighbors.
    ///
    /// If maximum_nprobes is set then this method will return the partitions
    /// that are most likely to contain the nearest neighbors (e.g. the closest
    /// partitions to the query vector).
    ///
    /// Return the partition ids and the distances between the query and the centroids,
    /// the results should be in sorted order from closest to farthest.
    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)>;

    /// Get the total number of partitions in the index.
    fn total_partitions(&self) -> usize;

    /// Search a single partition for nearest neighbors.
    ///
    /// This method should return the same results as [`VectorIndex::search`] method except
    /// that it will only search a single partition.
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// If the index is loadable by IVF, so it can be a sub-index that
    /// is loaded on demand by IVF.
    fn is_loadable(&self) -> bool;

    /// Use residual vector to search.
    fn use_residual(&self) -> bool;

    // async fn append(&self, batches: Vec<RecordBatch>) -> Result<()>;
    // async fn merge(&self, indices: Vec<Arc<dyn VectorIndex>>) -> Result<()>;

    /// Load the index from the reader on-demand.
    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>>;

    /// Load the partition from the reader on-demand.
    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        _partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        self.load(reader, offset, length).await
    }

    // for IVF only
    async fn partition_reader(
        &self,
        _partition_id: usize,
        _with_vector: bool,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!("only for IVF")
    }

    // for SubIndex only
    async fn to_batch_stream(&self, with_vector: bool) -> Result<SendableRecordBatchStream>;

    fn num_rows(&self) -> u64;

    /// Return the IDs of rows in the index.
    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_>;

    /// Remap the index according to mapping
    ///
    /// Each item in mapping describes an old row id -> new row id
    /// pair.  If old row id -> None then that row id has been
    /// deleted and can be removed from the index.
    ///
    /// If an old row id is not in the mapping then it should be
    /// left alone.
    async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()>;

    /// The metric type of this vector index.
    fn metric_type(&self) -> DistanceType;

    fn ivf_model(&self) -> &IvfModel;
    fn quantizer(&self) -> Quantizer;
    fn partition_size(&self, part_id: usize) -> usize;

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType);

    /// Export the index state needed for reconstruction from a disk cache.
    /// Returns `None` if this index type doesn't support persistent caching.
    fn cacheable_state(&self) -> Option<Box<dyn VectorIndexData>> {
        None
    }
}

// it can be an IVF index or a partition of IVF index
pub trait VectorIndexCacheEntry: Debug + Send + Sync + DeepSizeOf {
    fn as_any(&self) -> &dyn Any;
}
