// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_index::IndexType;
use lance_table::format::IndexMetadata;

use crate::{
    Error, Result,
    index::api::{IndexSegment, IndexSegmentPlan},
};

/// Plan physical segments for staged bitmap-index outputs.
///
/// Each staged bitmap root is already a complete canonical physical segment,
/// so planning preserves one output segment for each staged source segment.
pub(in crate::index) fn plan_segments(
    segments: &[IndexMetadata],
    target_segment_bytes: Option<u64>,
) -> Result<Vec<IndexSegmentPlan>> {
    if let Some(0) = target_segment_bytes {
        return Err(Error::invalid_input(
            "target_segment_bytes must be greater than zero".to_string(),
        ));
    }
    if target_segment_bytes.is_some() && segments.len() > 1 {
        return Err(Error::invalid_input(
            "Bitmap segment builder does not yet support merging multiple source segments"
                .to_string(),
        ));
    }

    segments
        .iter()
        .map(|segment| {
            let fragment_bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "Segment '{}' is missing fragment coverage",
                    segment.uuid
                ))
            })?;
            let index_details = segment.index_details.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "Segment '{}' is missing index details",
                    segment.uuid
                ))
            })?;
            if !index_details.type_url.ends_with("BitmapIndexDetails") {
                return Err(Error::index(format!(
                    "Segment '{}' is not a bitmap index segment",
                    segment.uuid
                )));
            }

            let built_segment = IndexSegment::new(
                segment.uuid,
                fragment_bitmap.iter(),
                index_details.clone(),
                segment.index_version,
            );
            let estimated_bytes = segment
                .files
                .as_ref()
                .map(|files| files.iter().map(|file| file.size_bytes).sum())
                .unwrap_or(0);
            Ok(IndexSegmentPlan::new(
                built_segment,
                vec![segment.clone()],
                estimated_bytes,
                Some(IndexType::Bitmap),
            ))
        })
        .collect()
}

/// Finalize one staged bitmap root into a commit-ready physical segment.
pub(in crate::index) async fn build_segment(
    segment_plan: &IndexSegmentPlan,
) -> Result<IndexSegment> {
    let built_segment = segment_plan.segment().clone();
    let source_segments = segment_plan.segments();
    if source_segments.len() != 1 {
        return Err(Error::invalid_input(
            "Bitmap segment builder does not yet support merging multiple source segments"
                .to_string(),
        ));
    }
    let source_segment = &source_segments[0];
    if source_segment.uuid != built_segment.uuid() {
        return Err(Error::invalid_input(
            "Bitmap segment builder requires the built segment UUID to match the staged source UUID"
                .to_string(),
        ));
    }

    Ok(built_segment)
}
