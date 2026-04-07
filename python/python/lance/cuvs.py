# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import json
import re
import tempfile
from importlib import import_module
from typing import TYPE_CHECKING, Iterator, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from .file import LanceFileSession
from .lance import PartitionArtifactBuilder
from .dependencies import numpy as np
from .log import LOGGER
from .util import _normalize_metric_type

if TYPE_CHECKING:
    from pathlib import Path

PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY = (
    "lance:index_build:precomputed_encoded_partition_sizes"
)
PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY = (
    "lance:index_build:precomputed_encoded_partition_fragment_ids"
)
PRECOMPUTED_ENCODED_TOTAL_LOSS_METADATA_KEY = (
    "lance:index_build:precomputed_encoded_total_loss"
)

PARTITION_ARTIFACT_MANIFEST_VERSION = 1
PARTITION_ARTIFACT_MANIFEST_FILE_NAME = "manifest.json"
PARTITION_ARTIFACT_METADATA_FILE_NAME = "metadata.lance"
PARTITION_ARTIFACT_PARTITIONS_DIR = "partitions"
DEFAULT_PARTITION_ARTIFACT_BUCKETS = 256
PARTITION_ARTIFACT_ROW_ID_COLUMN = "_rowid"

def build_vector_index_on_cuvs(
    dataset,
    column: str,
    metric_type: str,
    accelerator: str,
    num_partitions: int,
    num_sub_vectors: int,
    dst_dataset_uri: str | Path | None = None,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    batch_size: int = 1024 * 128,
    filter_nan: bool = True,
):
    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()

    trained_index, ivf_centroids, pq_codebook = _train_ivf_pq_index_on_cuvs(
        dataset,
        column,
        num_partitions,
        metric_type,
        accelerator,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
    )
    artifact_root, artifact_files = one_pass_assign_ivf_pq_on_cuvs(
        dataset,
        column,
        metric_type,
        accelerator,
        ivf_centroids,
        pq_codebook,
        trained_index=trained_index,
        dst_dataset_uri=dst_dataset_uri,
        batch_size=batch_size,
        filter_nan=filter_nan,
    )
    return artifact_root, artifact_files, ivf_centroids, pq_codebook


def is_cuvs_accelerator(accelerator: object) -> bool:
    return accelerator == "cuvs"


def _require_cuvs():
    try:
        return import_module("cuvs.neighbors.ivf_pq")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "accelerator='cuvs' requires cuVS Python bindings to be installed. "
            "Install a CUDA-matched package such as 'cuvs-cu12' or 'cuvs-cu13' "
            "from https://pypi.nvidia.com."
        ) from exc


def _optional_cupy():
    try:
        return import_module("cupy")
    except ModuleNotFoundError:
        return None


def _make_progress(total: int):
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total)
    except ModuleNotFoundError:

        class _NoOpProgress:
            def set_description(self, _description: str):
                return None

            def update(self, _count: int):
                return None

            def close(self):
                return None

        return _NoOpProgress()


def _metric_to_cuvs(metric_type: str) -> str:
    metric_type = _normalize_metric_type(metric_type).lower()
    if metric_type in {"l2", "euclidean"}:
        return "sqeuclidean"
    if metric_type == "dot":
        return "inner_product"
    if metric_type == "cosine":
        return "cosine"
    raise ValueError(f"Metric '{metric_type}' is not supported by cuVS IVF_PQ")


def _coerce_float_matrix(matrix: np.ndarray, *, column: str) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D training matrix for column '{column}', got {matrix.shape}"
        )
    if matrix.dtype == np.float64:
        matrix = matrix.astype(np.float32)
    elif matrix.dtype not in (np.float16, np.float32):
        matrix = matrix.astype(np.float32)
    return matrix


def _column_to_numpy(table: pa.Table | pa.RecordBatch, column: str) -> np.ndarray:
    array = table.column(column)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if len(array) == 0:
        raise ValueError("cuVS training requires at least one training vector")

    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        matrix = values.reshape(len(array), array.type.list_size)
        return _coerce_float_matrix(matrix, column=column)

    values = array.to_pylist()
    return _coerce_float_matrix(np.asarray(values), column=column)


def _annotate_precomputed_encoded_dataset(
    dataset,
    partition_sizes: list[int],
    *,
    total_loss: float | None = None,
) -> None:
    partition_fragments = [[] for _ in range(len(partition_sizes))]
    for fragment in dataset.get_fragments():
        fragment_partitions = set()
        scanner = fragment.scanner(columns=["__ivf_part_id"])
        for batch in scanner.to_batches():
            fragment_partitions.update(
                int(partition_id)
                for partition_id in np.unique(
                    batch.column("__ivf_part_id").to_numpy(zero_copy_only=False)
                )
            )
        for partition_id in fragment_partitions:
            partition_fragments[partition_id].append(int(fragment.metadata.id))

    metadata = {
        PRECOMPUTED_ENCODED_PARTITION_SIZES_METADATA_KEY: json.dumps(
            [int(size) for size in partition_sizes]
        ),
        PRECOMPUTED_ENCODED_PARTITION_FRAGMENT_IDS_METADATA_KEY: json.dumps(
            partition_fragments
        ),
    }
    if total_loss is not None:
        metadata[PRECOMPUTED_ENCODED_TOTAL_LOSS_METADATA_KEY] = json.dumps(
            float(total_loss)
        )
    dataset.update_metadata(metadata)


def _as_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like

    if hasattr(array_like, "copy_to_host"):
        return np.asarray(array_like.copy_to_host())

    try:
        array = np.asarray(array_like)
        if isinstance(array, np.ndarray):
            return array
    except Exception:
        pass

    if hasattr(array_like, "get"):
        return np.asarray(array_like.get())

    cupy = _optional_cupy()
    if cupy is not None:
        return cupy.asnumpy(array_like)

    raise TypeError("Unable to convert cuVS output to numpy")


def _normalize_artifact_root(path_or_uri: str | Path) -> str:
    root = str(path_or_uri)
    if re.search(r".:\\", root) is not None:
        root = root.replace("\\", "/", 1)
    return root


def _make_metadata_table(
    ivf_centroids: np.ndarray,
    pq_codebook: np.ndarray,
) -> pa.Table:
    dimension = ivf_centroids.shape[1]
    subvector_dim = pq_codebook.shape[2]
    ivf_type = pa.list_(pa.list_(pa.float32(), dimension))
    pq_type = pa.list_(pa.list_(pa.float32(), subvector_dim))
    ivf_values = pa.array([ivf_centroids.tolist()], type=ivf_type)
    pq_values = pa.array(
        [pq_codebook.reshape(-1, subvector_dim).tolist()],
        type=pq_type,
    )
    return pa.Table.from_arrays(
        [ivf_values, pq_values],
        names=["_ivf_centroids", "_pq_codebook"],
    )


def _write_partition_artifact_metadata(
    session: LanceFileSession,
    *,
    ivf_centroids: np.ndarray,
    pq_codebook: np.ndarray,
    metric_type: str,
    num_bits: int,
) -> None:
    metadata_table = _make_metadata_table(ivf_centroids, pq_codebook)
    with session.open_writer(
        PARTITION_ARTIFACT_METADATA_FILE_NAME,
        schema=metadata_table.schema,
        version="2.2",
    ) as writer:
        writer.add_schema_metadata("lance:index_build:artifact_version", "1")
        writer.add_schema_metadata(
            "lance:index_build:distance_type", _normalize_metric_type(metric_type)
        )
        writer.add_schema_metadata(
            "lance:index_build:num_partitions", str(ivf_centroids.shape[0])
        )
        writer.add_schema_metadata(
            "lance:index_build:num_sub_vectors", str(pq_codebook.shape[0])
        )
        writer.add_schema_metadata("lance:index_build:num_bits", str(num_bits))
        writer.add_schema_metadata("lance:index_build:dimension", str(ivf_centroids.shape[1]))
        writer.write_batch(metadata_table)


def _write_partition_artifact(
    batches: Iterator[pa.RecordBatch],
    *,
    artifact_root: str | Path,
    ivf_centroids: np.ndarray,
    pq_codebook: np.ndarray,
    metric_type: str,
    num_bits: int,
    num_partitions: int,
    total_loss: float | None = None,
) -> tuple[str, list[str]]:
    artifact_root = _normalize_artifact_root(artifact_root)
    session = LanceFileSession(artifact_root)
    builder = PartitionArtifactBuilder(
        artifact_root,
        num_partitions=num_partitions,
        pq_code_width=pq_codebook.shape[0],
    )
    for batch in batches:
        builder.append_batch(batch)

    _write_partition_artifact_metadata(
        session,
        ivf_centroids=ivf_centroids,
        pq_codebook=pq_codebook,
        metric_type=metric_type,
        num_bits=num_bits,
    )
    artifact_files = builder.finish(
        PARTITION_ARTIFACT_METADATA_FILE_NAME,
        float(total_loss) if total_loss is not None else None,
    )
    artifact_files.insert(1, PARTITION_ARTIFACT_METADATA_FILE_NAME)
    return artifact_root, artifact_files


def _to_cuvs_transform_input(matrix: np.ndarray):
    cupy = _optional_cupy()
    if cupy is None:
        raise ModuleNotFoundError(
            "accelerator='cuvs' full index build requires the 'cupy' package "
            "to pass transform batches in device memory"
        )
    return cupy.asarray(matrix)


def _normalize_centroids(index, num_partitions: int, dimension: int) -> np.ndarray:
    centroids = _as_numpy(index.centers)
    if centroids.shape != (num_partitions, dimension):
        raise ValueError(
            "cuVS returned incompatible IVF centroids shape: "
            f"expected {(num_partitions, dimension)}, got {centroids.shape}"
        )
    return centroids


def _normalize_pq_codebook(
    index, num_sub_vectors: int, num_bits: int, dimension: int
) -> np.ndarray:
    pq_book_size = 1 << num_bits
    subvector_dim = dimension // num_sub_vectors
    pq_centers = _as_numpy(index.pq_centers)

    expected_shapes = {
        (subvector_dim, num_sub_vectors, pq_book_size): (1, 2, 0),
        (num_sub_vectors, subvector_dim, pq_book_size): (0, 2, 1),
        (num_sub_vectors, pq_book_size, subvector_dim): None,
    }
    transpose = expected_shapes.get(pq_centers.shape)
    if transpose is None and pq_centers.shape not in expected_shapes:
        raise ValueError(
            "cuVS returned incompatible PQ codebook shape: expected one of "
            f"{list(expected_shapes.keys())}, got {pq_centers.shape}"
        )
    if transpose is not None:
        pq_centers = np.transpose(pq_centers, transpose)
    return pq_centers


def _estimate_trainset_fraction(
    num_rows: int, num_partitions: int, sample_rate: int
) -> float:
    if num_rows <= 0:
        raise ValueError("cuVS training requires a non-empty dataset")
    desired_rows = max(num_partitions * sample_rate, 256 * 256)
    return min(1.0, desired_rows / num_rows)


def _sample_training_table(
    dataset, column: str, train_rows: int, filt: str | None
) -> pa.Table:
    if filt is None:
        return dataset.sample(train_rows, columns=[column], randomize_order=True)

    total_rows = dataset.count_rows()
    sample_rows = min(total_rows, max(train_rows * 2, train_rows + 1024))
    trainset = dataset.sample(sample_rows, columns=[column], randomize_order=True)
    trainset = trainset.filter(pc.is_valid(trainset.column(column)))
    if len(trainset) >= train_rows or sample_rows == total_rows:
        return trainset.slice(0, min(train_rows, len(trainset)))

    return dataset.to_table(columns=[column], filter=filt, limit=train_rows)


def _train_ivf_pq_index_on_cuvs(
    dataset,
    column: str,
    num_partitions: int,
    metric_type: str,
    accelerator: str,
    num_sub_vectors: int,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    filter_nan: bool = True,
):
    if accelerator != "cuvs":
        raise ValueError("cuVS acceleration only supports accelerator='cuvs'")
    if num_bits != 8:
        raise ValueError("cuVS IVF_PQ integration currently supports only num_bits=8")

    dimension = dataset.schema.field(column).type.list_size
    if dimension % num_sub_vectors != 0:
        raise ValueError(
            "cuVS IVF_PQ integration requires vector dimension to be divisible by "
            "num_sub_vectors"
        )

    if dataset.schema.field(column).nullable and filter_nan:
        filt = f"{column} is not null"
    else:
        filt = None

    num_rows = dataset.count_rows(filter=filt)
    if num_rows == 0:
        raise ValueError("cuVS training requires at least one non-null training vector")

    train_rows = max(1, min(num_rows, max(num_partitions * sample_rate, 256 * 256)))
    trainset = _sample_training_table(dataset, column, train_rows, filt)
    matrix = _column_to_numpy(trainset, column)

    ivf_pq = _require_cuvs()
    build_params = ivf_pq.IndexParams(
        n_lists=num_partitions,
        metric=_metric_to_cuvs(metric_type),
        kmeans_n_iters=max_iters,
        kmeans_trainset_fraction=_estimate_trainset_fraction(
            matrix.shape[0], num_partitions, sample_rate
        ),
        pq_bits=num_bits,
        pq_dim=num_sub_vectors,
        codebook_kind="subspace",
        force_random_rotation=False,
        add_data_on_build=False,
    )

    index = ivf_pq.build(build_params, matrix)
    centroids = _normalize_centroids(index, num_partitions, dimension)
    pq_codebook = _normalize_pq_codebook(index, num_sub_vectors, num_bits, dimension)
    return index, centroids, pq_codebook


def one_pass_assign_ivf_pq_on_cuvs(
    dataset,
    column: str,
    metric_type: str,
    accelerator: str,
    ivf_centroids: np.ndarray,
    pq_codebook: np.ndarray,
    trained_index=None,
    dst_dataset_uri: str | Path | None = None,
    batch_size: int = 1024 * 128,
    *,
    filter_nan: bool = True,
):
    if accelerator != "cuvs":
        raise ValueError("cuVS acceleration only supports accelerator='cuvs'")

    num_rows = dataset.count_rows()
    if dataset.schema.field(column).nullable and filter_nan:
        filt = f"{column} is not null"
    else:
        filt = None

    num_sub_vectors = pq_codebook.shape[0]
    ivf_pq = _require_cuvs()

    if trained_index is None:
        raise ValueError(
            "one_pass_assign_ivf_pq_on_cuvs requires a trained cuVS index for "
            "single-node transform"
        )
    transform_code_width = (trained_index.pq_dim * trained_index.pq_bits + 7) // 8
    if transform_code_width != num_sub_vectors:
        raise ValueError(
            "cuVS transform output is incompatible with Lance IVF_PQ for this "
            "configuration: expected "
            f"{num_sub_vectors} PQ code columns, but cuVS will produce "
            f"{transform_code_width}. Use a configuration where "
            "ceil(pq_dim * pq_bits / 8) == num_sub_vectors."
        )

    progress = _make_progress(num_rows)
    progress.set_description("Assigning partitions and computing pq codes")
    num_partitions = ivf_centroids.shape[0]
    partition_sizes = np.zeros(num_partitions, dtype=np.int64)

    output_schema = pa.schema(
        [
            pa.field(PARTITION_ARTIFACT_ROW_ID_COLUMN, pa.uint64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__pq_code", pa.list_(pa.uint8(), list_size=num_sub_vectors)),
        ]
    )

    def _partition_and_pq_codes_assignment() -> Iterator[pa.RecordBatch]:
        for batch in dataset.to_batches(
            columns=[column],
            filter=filt,
            with_row_id=True,
            batch_size=batch_size,
        ):
            vectors = _column_to_numpy(batch, column)
            row_ids = batch.column("_rowid").to_numpy()
            valid_mask = np.isfinite(vectors).all(axis=1)
            if not np.all(valid_mask):
                LOGGER.warning(
                    "%s vectors are ignored during partition assignment",
                    len(valid_mask) - int(valid_mask.sum()),
                )
                row_ids = row_ids[valid_mask]
                vectors = vectors[valid_mask]
            if len(row_ids) == 0:
                continue
            partitions, pq_codes = ivf_pq.transform(
                trained_index, _to_cuvs_transform_input(vectors)
            )
            partitions = _as_numpy(partitions).astype(np.uint32, copy=False)
            partition_sizes[:] += np.bincount(partitions, minlength=num_partitions)
            pq_codes = _as_numpy(pq_codes).astype(np.uint8, copy=False)
            if pq_codes.shape != (len(row_ids), num_sub_vectors):
                raise ValueError(
                    "cuVS transform returned incompatible PQ codes shape: "
                    f"expected {(len(row_ids), num_sub_vectors)}, got {pq_codes.shape}"
                )

            pq_values = pa.array(pq_codes.reshape(-1), type=pa.uint8())
            pq_code_array = pa.FixedSizeListArray.from_arrays(
                pq_values, num_sub_vectors
            )
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(partitions, type=pa.uint32()),
                    pq_code_array,
                ],
                schema=output_schema,
            )
            progress.update(len(row_ids))

    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()
    artifact_root, artifact_files = _write_partition_artifact(
        _partition_and_pq_codes_assignment(),
        artifact_root=dst_dataset_uri,
        ivf_centroids=ivf_centroids,
        pq_codebook=pq_codebook,
        metric_type=metric_type,
        num_bits=8,
        num_partitions=num_partitions,
    )

    progress.close()
    LOGGER.info("Saved precomputed partition artifact to %s", artifact_root)
    return str(artifact_root), artifact_files


def train_ivf_pq_on_cuvs(
    dataset,
    column: str,
    num_partitions: int,
    metric_type: str,
    accelerator: str,
    num_sub_vectors: int,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    filter_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    _, centroids, pq_codebook = _train_ivf_pq_index_on_cuvs(
        dataset,
        column,
        num_partitions,
        metric_type,
        accelerator,
        num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
    )
    return centroids, pq_codebook


def one_pass_train_ivf_pq_on_cuvs(
    dataset,
    column: str,
    num_partitions: int,
    metric_type: str,
    accelerator: str,
    num_sub_vectors: int,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    filter_nan: bool = True,
):
    return train_ivf_pq_on_cuvs(
        dataset,
        column,
        num_partitions,
        metric_type,
        accelerator,
        num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
    )


def prepare_global_ivf_pq_on_cuvs(
    dataset,
    column: str,
    num_partitions: int,
    num_sub_vectors: int,
    *,
    distance_type: str = "l2",
    accelerator: str = "cuvs",
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
):
    centroids, pq_codebook = train_ivf_pq_on_cuvs(
        dataset,
        column,
        num_partitions,
        distance_type,
        accelerator,
        num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
    )
    return {"ivf_centroids": centroids, "pq_codebook": pq_codebook}
