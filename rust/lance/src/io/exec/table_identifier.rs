// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helpers for converting between [`Dataset`] and [`TableIdentifier`](pb::TableIdentifier) proto.
//!
//! These are used by multiple proto modules (`filtered_read_proto`, `ann_ivf_proto`)
//! to identify a dataset for remote reconstruction.

use std::sync::Arc;

use lance_core::Result;
use lance_datafusion::pb;
use lance_io::object_store::StorageOptions;
use prost::Message;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;

/// Build a [`TableIdentifier`] from a [`Dataset`].
///
/// Default: lightweight mode (uri + version + etag only, no serialized manifest).
/// Includes the dataset's latest storage options (if any) so the remote executor
/// can open or cache the dataset with the correct storage configuration.
pub async fn table_identifier_from_dataset(dataset: &Dataset) -> Result<pb::TableIdentifier> {
    Ok(pb::TableIdentifier {
        uri: dataset.uri().to_string(),
        version: dataset.manifest.version,
        manifest_etag: dataset.manifest_location.e_tag.clone(),
        serialized_manifest: None,
        storage_options: dataset
            .latest_storage_options()
            .await?
            .map(|StorageOptions(m)| m)
            .unwrap_or_default(),
    })
}

/// Build a [`TableIdentifier`] with serialized manifest bytes included.
///
/// Fast path: remote executor skips manifest read from storage.
pub async fn table_identifier_from_dataset_with_manifest(
    dataset: &Dataset,
) -> Result<pb::TableIdentifier> {
    let manifest_proto = lance_table::format::pb::Manifest::from(dataset.manifest.as_ref());
    Ok(pb::TableIdentifier {
        uri: dataset.uri().to_string(),
        version: dataset.manifest.version,
        manifest_etag: dataset.manifest_location.e_tag.clone(),
        serialized_manifest: Some(manifest_proto.encode_to_vec()),
        storage_options: dataset
            .latest_storage_options()
            .await?
            .map(|StorageOptions(m)| m)
            .unwrap_or_default(),
    })
}

/// Open a dataset from a table identifier proto.
pub async fn open_dataset_from_table_identifier(
    table_id: &pb::TableIdentifier,
) -> Result<Arc<Dataset>> {
    let mut builder = DatasetBuilder::from_uri(&table_id.uri).with_version(table_id.version);
    if let Some(manifest_bytes) = &table_id.serialized_manifest {
        builder = builder.with_serialized_manifest(manifest_bytes)?;
    }
    if !table_id.storage_options.is_empty() {
        builder = builder.with_storage_options(table_id.storage_options.clone());
    }
    Ok(Arc::new(builder.load().await?))
}
