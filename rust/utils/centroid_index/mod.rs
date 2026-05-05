//! Configurable IVF centroid probe index (dense matmul vs CAGRA graph).

#[cfg(feature = "cagra")]
mod hnsw_backend;

use anyhow::{Result, anyhow};
#[cfg(feature = "cagra")]
use anyhow::Context;
use tch::{Device, Tensor};

#[cfg(feature = "cagra")]
use std::sync::Arc;

/// Build-time / search-time parameters for a CAGRA index over centroid rows.
#[derive(Debug, Clone, Copy)]
pub struct HnswBuildParams {
    /// Graph `M` — max neighbors per vertex (CAGRA `graph_degree`).
    pub m: u32,
    /// `intermediate_graph_degree` during graph build.
    pub ef_construction: u32,
    /// `itopk_size` / search beam width during querying.
    pub ef_search: u32,
}

impl Default for HnswBuildParams {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 40,
            ef_search: 64,
        }
    }
}

/// Resolved configuration for the centroid index. Constructed from a
/// user-provided kind string + an optional parameter dict at index load
/// time, then passed to [`CentroidIndex::build`].
#[derive(Debug, Clone, Copy)]
pub enum CentroidIndexConfig {
    Dense,
    Hnsw {
        params: HnswBuildParams,
    },
}

impl Default for CentroidIndexConfig {
    fn default() -> Self {
        CentroidIndexConfig::Dense
    }
}

/// Names of the parameter keys recognized for graph backends (`cagra`).
pub const HNSW_PARAM_KEYS: &[&str] = &[
    "graph_degree",
    "intermediate_graph_degree",
    "itopk_size",
];

/// A single user-provided parameter value. The pyfunction layer extracts
/// each entry of the parameter dict into one of these variants so the
/// core parsing logic stays free of pyo3 types.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i64),
    Str(String),
}

impl ParamValue {
    fn as_u32(&self, key: &str) -> Result<u32> {
        match self {
            ParamValue::Int(v) => {
                if *v < 0 {
                    Err(anyhow!(
                        "centroid_index param '{}' must be non-negative, got {}",
                        key,
                        v
                    ))
                } else if *v > u32::MAX as i64 {
                    Err(anyhow!(
                        "centroid_index param '{}' exceeds u32::MAX",
                        key
                    ))
                } else {
                    Ok(*v as u32)
                }
            }
            ParamValue::Str(_) => Err(anyhow!(
                "centroid_index param '{}' must be an integer",
                key
            )),
        }
    }
}

impl CentroidIndexConfig {
    /// Resolve a kind string and an optional parameter map into a config.
    ///
    /// * `kind`: case-insensitive `"dense"` (or `"brute"`) or `"cagra"`. `None` selects
    ///   the default.
    /// * `params`: only meaningful for graph backends. Unknown keys are
    ///   rejected. For dense, any params raise an error.
    pub fn parse<'a, I>(kind: Option<&str>, params: Option<I>) -> Result<Self>
    where
        I: IntoIterator<Item = (&'a str, ParamValue)>,
    {
        let kind = kind.unwrap_or("dense").to_ascii_lowercase();
        match kind.as_str() {
            "dense" | "brute" | "bruteforce" | "brute_force" => {
                if params.is_some() {
                    let collected: Vec<_> = params.unwrap().into_iter().collect();
                    if !collected.is_empty() {
                        return Err(anyhow!(
                            "centroid_index_params is only valid for centroid_index='cagra'; \
                             got {} entries with centroid_index='dense'",
                            collected.len()
                        ));
                    }
                }
                Ok(Self::Dense)
            }
            "cagra" => {
                let mut hp = HnswBuildParams::default();
                if let Some(iter) = params {
                    for (raw_key, value) in iter {
                        let key = raw_key.to_ascii_lowercase();
                        match key.as_str() {
                            "graph_degree" => hp.m = value.as_u32(&key)?,
                            "intermediate_graph_degree" => {
                                hp.ef_construction = value.as_u32(&key)?
                            }
                            "itopk_size" => hp.ef_search = value.as_u32(&key)?,
                            other => {
                                return Err(anyhow!(
                                    "unknown centroid_index param '{}'; expected one of {:?}",
                                    other,
                                    HNSW_PARAM_KEYS
                                ));
                            }
                        }
                    }
                }
                if hp.m < 2 {
                    return Err(anyhow!(
                        "CAGRA parameter graph_degree must be >= 2, got {}",
                        hp.m
                    ));
                }
                if hp.ef_construction < 1 {
                    return Err(anyhow!(
                        "intermediate_graph_degree must be >= 1, got {}",
                        hp.ef_construction
                    ));
                }
                if hp.ef_search < 1 {
                    return Err(anyhow!(
                        "itopk_size must be >= 1, got {}",
                        hp.ef_search
                    ));
                }
                Ok(Self::Hnsw { params: hp })
            }
            other => Err(anyhow!(
                "unknown centroid_index kind '{}': expected 'dense' or 'cagra'",
                other
            )),
        }
    }
}

/// Exact brute IVF top-`k`: `centroids @ q.T`.
pub(crate) fn dense_like_topk(centroids: &Tensor, queries: &Tensor, k: i64) -> (Tensor, Tensor) {
    let scores = centroids.matmul(&queries.transpose(0, 1));
    let (vals, ids) = scores.topk(k, 0, true, false);
    (
        ids.transpose(0, 1).contiguous(),
        vals.transpose(0, 1).contiguous(),
    )
}

#[cfg(not(feature = "cagra"))]
fn centroid_index_try_build_hnsw(_params: HnswBuildParams, _centroids: Tensor) -> Result<CentroidIndex> {
    Err(anyhow!(
        "centroid_index='cagra' requires rebuilding fast_plaid_rust with the Cargo feature `cagra` \
         (links cuVS). Pass e.g. `maturin develop --features cagra`."
    ))
}

#[cfg(feature = "cagra")]
fn centroid_index_try_build_hnsw(params: HnswBuildParams, centroids: Tensor) -> Result<CentroidIndex> {
    let hnsw = hnsw_backend::HnswCentroidState::build_centroids(&centroids, &params).with_context(
        || "failed to build CAGRA centroid index — ensure cuVS is installed and CUDA is available",
    )?;

    Ok(CentroidIndex::Hnsw {
        centroids,
        hnsw,
    })
}

/// Abstraction over the centroid lookup used during search.
///
/// Two operations are needed during a search:
///
/// 1. Pick the top-`k` centroids per query token (cell selection /
///    candidate generation).
/// 2. Score an arbitrary set of centroid IDs — produced by quantizing
///    document tokens — against the query (approximate MaxSim).
///
/// Both operations are batched over query tokens. The dense backend
/// computes `centroids @ q.T` directly; the HNSW backend runs a Faiss graph
/// search over centroid rows (CPU) and recomputes dot-product scores on the
/// centroid tensor.
pub enum CentroidIndex {
    /// Brute-force backend: holds the centroid matrix and computes
    /// `centroids @ q.T` on demand.
    Dense { centroids: Tensor },
    #[cfg(feature = "cagra")]
    /// Graph ANN probe selection (`topk`); same `score()` / `masked_topk` as dense.
    Hnsw {
        centroids: Tensor,
        hnsw: Arc<hnsw_backend::HnswCentroidState>,
    },
}

impl CentroidIndex {
    pub fn dense(centroids: Tensor) -> Self {
        CentroidIndex::Dense { centroids }
    }

    pub fn build(
        config: CentroidIndexConfig,
        centroids: Tensor,
        _codec_device: Device,
    ) -> Result<Self> {
        match config {
            CentroidIndexConfig::Dense => Ok(Self::dense(centroids)),
            CentroidIndexConfig::Hnsw { params } => centroid_index_try_build_hnsw(params, centroids),
        }
    }

    /// Top-`k` IVF centroids per query token (`score` maximization equals L2-min among
    /// L2-normalized centroid rows). Uses HNSW when configured.
    pub fn topk(&self, queries: &Tensor, k: i64) -> Result<(Tensor, Tensor)> {
        match self {
            CentroidIndex::Dense { centroids } => Ok(dense_like_topk(centroids, queries, k)),
            #[cfg(feature = "cagra")]
            CentroidIndex::Hnsw {
                centroids,
                hnsw,
            } => hnsw.topk_centroids(centroids, queries, k),
        }
    }

    pub fn masked_topk(
        &self,
        queries: &Tensor,
        candidate_ids: &Tensor,
        k: i64,
    ) -> (Tensor, Tensor) {
        let scores = self.score(queries, candidate_ids);
        let (vals, local_ids) = scores.topk(k, 0, true, false);
        let flat_local = local_ids.flatten(0, -1);
        let global_flat = candidate_ids.index_select(0, &flat_local);
        let global_ids = global_flat.view_as(&local_ids).transpose(0, 1).contiguous();
        let scores_out = vals.transpose(0, 1).contiguous();
        (global_ids, scores_out)
    }

    pub fn score(&self, queries: &Tensor, centroid_ids: &Tensor) -> Tensor {
        match self {
            CentroidIndex::Dense { centroids } => centroids
                .index_select(0, centroid_ids)
                .matmul(&queries.transpose(0, 1)),
            #[cfg(feature = "cagra")]
            CentroidIndex::Hnsw { centroids, .. } => centroids
                .index_select(0, centroid_ids)
                .matmul(&queries.transpose(0, 1)),
        }
    }
}

impl Clone for CentroidIndex {
    fn clone(&self) -> Self {
        match self {
            CentroidIndex::Dense { centroids } => CentroidIndex::Dense {
                centroids: centroids.shallow_clone(),
            },
            #[cfg(feature = "cagra")]
            CentroidIndex::Hnsw { centroids, hnsw } => CentroidIndex::Hnsw {
                centroids: centroids.shallow_clone(),
                hnsw: Arc::clone(hnsw),
            },
        }
    }
}
