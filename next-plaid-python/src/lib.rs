use ::next_plaid::{filtering, IndexConfig, MmapIndex};
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pythonize::depythonize;

/// Create a PLAID index from document embeddings.
///
/// Args:
///     embeddings: List of 2D numpy arrays, each shape (num_tokens, embedding_dim).
///     index_path: Directory path where the index will be stored.
///     nbits: Quantization bits (2 or 4). Default: 4.
///     batch_size: Batch size for processing. Default: 50000.
///     kmeans_niters: Number of K-means iterations. Default: 4.
///     seed: Random seed for reproducibility. Default: None.
///     force_cpu: Skip CUDA even if available. Default: False.
#[pyfunction]
#[pyo3(signature = (embeddings, index_path, nbits=4, batch_size=50_000, kmeans_niters=4, seed=None, force_cpu=false))]
fn create_index(
    embeddings: Vec<PyReadonlyArray2<f32>>,
    index_path: &str,
    nbits: usize,
    batch_size: usize,
    kmeans_niters: usize,
    seed: Option<u64>,
    force_cpu: bool,
) -> PyResult<()> {
    let arrays: Vec<Array2<f32>> = embeddings
        .iter()
        .map(|e| e.as_array().to_owned())
        .collect();

    let config = IndexConfig {
        nbits,
        batch_size,
        kmeans_niters,
        seed,
        force_cpu,
        ..Default::default()
    };

    MmapIndex::create_with_kmeans(&arrays, index_path, &config)
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
