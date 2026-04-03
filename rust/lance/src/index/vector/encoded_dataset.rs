// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::Fields;
use futures::StreamExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::v3::shuffler::ShuffleReader;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_table::format::Fragment;
use log::warn;
use serde::de::DeserializeOwned;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;

pub(crate) const PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY: &str =
    "lance:index_build:precomputed_encoded_partition_sizes";
pub(crate) const PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY: &str =
    "lance:index_build:precomputed_encoded_partition_fragment_ids";
pub(crate) const PRECOMPUTED_ENCODED_TOTAL_LOSS_METADATA_KEY: &str =
    "lance:index_build:precomputed_encoded_total_loss";

const PRECOMPUTED_ROW_ID_COLUMN: &str = "row_id";

pub(crate) struct EncodedDatasetShuffleReader {
    dataset: Dataset,
    row_id_column: String,
    partition_sizes: Vec<usize>,
    partition_fragments: Option<Vec<Vec<Fragment>>>,
    total_loss: Option<f64>,
}

impl EncodedDatasetShuffleReader {
    pub(crate) async fn try_open(
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(storage_options) = storage_options {
            builder = builder.with_storage_options(storage_options.clone());
        }
        let dataset = builder.load().await?;
        Self::try_new(dataset)
    }

    pub(crate) fn try_new(dataset: Dataset) -> Result<Self> {
        let row_id_column = if dataset.schema().field(ROW_ID).is_some() {
            ROW_ID.to_string()
        } else if dataset.schema().field(PRECOMPUTED_ROW_ID_COLUMN).is_some() {
            PRECOMPUTED_ROW_ID_COLUMN.to_string()
        } else {
            return Err(Error::invalid_input(format!(
                "precomputed encoded dataset must contain '{}' or '{}' column",
                ROW_ID, PRECOMPUTED_ROW_ID_COLUMN
            )));
        };

        for required_column in [PART_ID_COLUMN, PQ_CODE_COLUMN] {
            if dataset.schema().field(required_column).is_none() {
                return Err(Error::invalid_input(format!(
                    "precomputed encoded dataset is missing required column '{}'",
                    required_column
                )));
            }
        }

        let metadata = dataset.metadata();
        let partition_sizes: Vec<usize> =
            parse_required_metadata(metadata, PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY)?;

        let partition_fragments = parse_optional_metadata::<Vec<Vec<u64>>>(
            metadata,
            PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY,
        )?
        .map(|partition_fragment_ids| resolve_partition_fragments(&dataset, partition_fragment_ids))
        .transpose()?;

        if let Some(partition_fragments) = partition_fragments.as_ref() {
            if partition_fragments.len() != partition_sizes.len() {
                return Err(Error::invalid_input(format!(
                    "metadata '{}' has {} partitions but '{}' has {}",
                    PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY,
                    partition_fragments.len(),
                    PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY,
                    partition_sizes.len(),
                )));
            }
        }

        let total_loss =
            parse_optional_metadata::<f64>(metadata, PRECOMPUTED_ENCODED_TOTAL_LOSS_METADATA_KEY)?;

        Ok(Self {
            dataset,
            row_id_column,
            partition_sizes,
            partition_fragments,
            total_loss,
        })
    }

    fn rename_row_id(
        stream: impl RecordBatchStream + Unpin + 'static,
        row_id_idx: usize,
    ) -> impl RecordBatchStream + Unpin + 'static {
        let new_schema = Arc::new(arrow_schema::Schema::new(
            stream
                .schema()
                .fields
                .iter()
                .enumerate()
                .map(|(field_idx, field)| {
                    if field_idx == row_id_idx {
                        arrow_schema::Field::new(
                            ROW_ID,
                            field.data_type().clone(),
                            field.is_nullable(),
                        )
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Fields>(),
        ));
        RecordBatchStreamAdapter::new(
            new_schema.clone(),
            stream.map(move |batch| match batch {
                Ok(batch) => {
                    arrow_array::RecordBatch::try_new(new_schema.clone(), batch.columns().to_vec())
                        .map_err(Error::from)
                }
                Err(error) => Err(error),
            }),
        )
    }
}

#[async_trait::async_trait]
impl ShuffleReader for EncodedDatasetShuffleReader {
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        if partition_id >= self.partition_sizes.len() {
            return Ok(None);
        }
        if self.partition_sizes[partition_id] == 0 {
            return Ok(None);
        }

        let mut scanner = self.dataset.scan();
        scanner.batch_readahead(get_num_compute_intensive_cpus());
        scanner.project(&[self.row_id_column.as_str(), PART_ID_COLUMN, PQ_CODE_COLUMN])?;

        if let Some(partition_fragments) = self.partition_fragments.as_ref() {
            let fragments = &partition_fragments[partition_id];
            if fragments.is_empty() {
                warn!(
                    "precomputed encoded dataset metadata has no fragments for non-empty partition {}, falling back to filtered scan",
                    partition_id
                );
            } else {
                scanner.with_fragments(fragments.clone());
            }
        }

        scanner.filter(&format!("{PART_ID_COLUMN} = {partition_id}"))?;
        let stream = scanner.try_into_stream().await?;
        if let Some((row_id_idx, _)) = stream.schema().column_with_name(PRECOMPUTED_ROW_ID_COLUMN) {
            Ok(Some(Box::new(Self::rename_row_id(stream, row_id_idx))))
        } else {
            Ok(Some(Box::new(stream)))
        }
    }

    fn partition_size(&self, partition_id: usize) -> Result<usize> {
        Ok(self.partition_sizes.get(partition_id).copied().unwrap_or(0))
    }

    fn total_loss(&self) -> Option<f64> {
        self.total_loss
    }
}

fn parse_required_metadata<T: DeserializeOwned>(
    metadata: &HashMap<String, String>,
    key: &str,
) -> Result<T> {
    let value = metadata.get(key).ok_or_else(|| {
        Error::invalid_input(format!(
            "precomputed encoded dataset is missing required metadata '{}'",
            key
        ))
    })?;
    parse_metadata_value(value, key)
}

fn parse_optional_metadata<T: DeserializeOwned>(
    metadata: &HashMap<String, String>,
    key: &str,
) -> Result<Option<T>> {
    metadata
        .get(key)
        .map(|value| parse_metadata_value(value, key))
        .transpose()
}

fn parse_metadata_value<T: DeserializeOwned>(value: &str, key: &str) -> Result<T> {
    serde_json::from_str(value).map_err(|error| {
        Error::invalid_input(format!(
            "failed to parse precomputed encoded dataset metadata '{}' from '{}': {}",
            key, value, error
        ))
    })
}

fn resolve_partition_fragments(
    dataset: &Dataset,
    partition_fragment_ids: Vec<Vec<u64>>,
) -> Result<Vec<Vec<Fragment>>> {
    let fragments_by_id = dataset
        .fragments()
        .iter()
        .cloned()
        .map(|fragment| (fragment.id, fragment))
        .collect::<HashMap<_, _>>();

    partition_fragment_ids
        .into_iter()
        .map(|fragment_ids| {
            fragment_ids
                .into_iter()
                .map(|fragment_id| {
                    fragments_by_id.get(&fragment_id).cloned().ok_or_else(|| {
                        Error::invalid_input(format!(
                            "precomputed encoded dataset metadata references unknown fragment id {}",
                            fragment_id
                        ))
                    })
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        FixedSizeListArray, RecordBatch, RecordBatchIterator, UInt8Array, UInt32Array, UInt64Array,
        cast::AsArray,
    };
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;

    use crate::dataset::WriteParams;

    #[tokio::test]
    async fn encoded_dataset_reader_reads_mapped_fragments_and_renames_row_id() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("row_id", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(PART_ID_COLUMN, arrow_schema::DataType::UInt32, false),
            arrow_schema::Field::new(
                PQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::UInt8,
                        true,
                    )),
                    2,
                ),
                true,
            ),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 11])),
                Arc::new(UInt32Array::from(vec![0_u32, 1])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![1, 2, 3, 4]), 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![12_u64, 13])),
                Arc::new(UInt32Array::from(vec![1_u32, 1])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![5, 6, 7, 8]), 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch1), Ok(batch2)], schema);
        let write_params = WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        };
        let mut dataset = Dataset::write(
            reader,
            "memory://precomputed-encoded-reader",
            Some(write_params),
        )
        .await
        .unwrap();

        let fragment_ids = dataset
            .get_fragments()
            .into_iter()
            .map(|fragment| fragment.metadata().id)
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids.len(), 2);

        dataset
            .update_metadata(vec![
                (
                    PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY.to_string(),
                    serde_json::to_string(&vec![1_usize, 3]).unwrap(),
                ),
                (
                    PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY.to_string(),
                    serde_json::to_string(&vec![
                        vec![fragment_ids[0] as u64],
                        vec![fragment_ids[0] as u64, fragment_ids[1] as u64],
                    ])
                    .unwrap(),
                ),
            ])
            .await
            .unwrap();

        let reader = EncodedDatasetShuffleReader::try_new(dataset).unwrap();
        assert_eq!(reader.partition_size(0).unwrap(), 1);
        assert_eq!(reader.partition_size(1).unwrap(), 3);

        let stream = reader.read_partition(1).await.unwrap().unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let row_ids = batches
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![11, 12, 13]);
        assert!(
            batches
                .iter()
                .all(|batch| batch.column_by_name("row_id").is_none())
        );
    }
}
