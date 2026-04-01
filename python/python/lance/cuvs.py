# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from importlib import import_module
from typing import Tuple

import pyarrow as pa
import pyarrow.compute as pc

from .dependencies import numpy as np


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


def _metric_to_cuvs(metric_type: str) -> str:
    metric_type = metric_type.lower()
    if metric_type in {"l2", "euclidean"}:
        return "sqeuclidean"
    if metric_type == "dot":
        return "inner_product"
    if metric_type == "cosine":
        return "cosine"
    raise ValueError(f"Metric '{metric_type}' is not supported by cuVS IVF_PQ")


def _column_to_numpy(table: pa.Table, column: str) -> np.ndarray:
    array = table.column(column).combine_chunks()
    values = array.to_pylist()
    if len(values) == 0:
        raise ValueError("cuVS training requires at least one training vector")
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D training matrix for column '{column}', got {matrix.shape}"
        )
    if matrix.dtype == np.float64:
        matrix = matrix.astype(np.float32)
    elif matrix.dtype not in (np.float16, np.float32):
        matrix = matrix.astype(np.float32)
    return matrix


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
