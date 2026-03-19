use ::next_plaid::index::create_index_files;
use ::next_plaid::{filtering, IndexConfig};
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pythonize::depythonize;

/// Create a PLAID index from document embeddings and pre-computed centroids.
///
/// K-means should be done externally (e.g. with fastkmeans in Python).
/// This function handles quantization, residual computation, and index file creation.
///
/// Args:
///     embeddings: List of 2D numpy arrays, each shape (num_tokens, embedding_dim).
///     centroids: 2D numpy array of shape (num_centroids, embedding_dim).
///     index_path: Directory path where the index will be stored.
///     nbits: Quantization bits (2 or 4). Default: 4.
///     batch_size: Batch size for processing. Default: 50000.
///     seed: Random seed for reproducibility. Default: None.
///     force_cpu: Skip CUDA for residual computation. Default: False.
#[pyfunction]
#[pyo3(signature = (embeddings, centroids, index_path, nbits=4, batch_size=50_000, seed=None, force_cpu=false))]
fn create_index(
    embeddings: Vec<PyReadonlyArray2<f32>>,
    centroids: PyReadonlyArray2<f32>,
    index_path: &str,
    nbits: usize,
    batch_size: usize,
    seed: Option<u64>,
    force_cpu: bool,
) -> PyResult<()> {
    let arrays: Vec<Array2<f32>> = embeddings
        .iter()
        .map(|e| e.as_array().to_owned())
        .collect();

    let centroids_array: Array2<f32> = centroids.as_array().to_owned();

    let config = IndexConfig {
        nbits,
        batch_size,
        seed,
        force_cpu,
        ..Default::default()
    };

    create_index_files(&arrays, centroids_array, index_path, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(())
}

/// Create a SQLite metadata database for the index.
///
/// Args:
///     index_path: Directory path of the index.
///     metadata: List of dicts, one per document. Values must be JSON-serializable.
///     doc_ids: List of document IDs matching the metadata entries.
///
/// Returns:
///     Number of metadata rows inserted.
#[pyfunction]
fn create_metadata(
    index_path: &str,
    metadata: Vec<Bound<'_, PyAny>>,
    doc_ids: Vec<i64>,
) -> PyResult<usize> {
    let values: Vec<serde_json::Value> = metadata
        .iter()
        .map(|obj| depythonize(obj))
        .collect::<Result<_, _>>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    filtering::create(index_path, &values, &doc_ids)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule(name = "next_plaid")]
fn next_plaid_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_index, m)?)?;
    m.add_function(wrap_pyfunction!(create_metadata, m)?)?;
    Ok(())
}
