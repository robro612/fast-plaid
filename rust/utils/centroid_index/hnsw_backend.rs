//! CAGRA index over IVF centroids (GPU; via cuVS).
//!
//! Neighbor IDs come from an approximate graph search among centroids using the
//! cuVS CAGRA index. Returned dot-product scores are recomputed on the centroid
//! tensor to stay consistent with the dense backend.
#![cfg(feature = "cagra")]

use std::sync::Arc;
use std::ops::Mul;

use anyhow::{anyhow, Context, Result};
use crate::utils::cagra_metric::cagra_index_params_set_cosine_expanded;
use crate::utils::tensor_finite::ensure_tensor_all_finite;
use cuvs::cagra::{Index, IndexParams, SearchAlgo, SearchParams};
use cuvs::{ManagedTensor, Resources};
use ndarray::Array2;
use parking_lot::Mutex;
use tch::{Device, Kind, Tensor, TchError};
use std::sync::atomic::{AtomicU64, Ordering};

/// cuVS CAGRA search kernel selection for centroid probing.
///
/// `AUTO` vs `MULTI_CTA` has **not** been stable across libcuvs builds / GPUs in the field:
/// one environment reportedly failed with `AUTO`, another failed with forced `MULTI_CTA` +
/// tuned block/team settings. This is controlled at runtime via
/// **`FASTPLAID_CAGRA_CENTROID_SEARCH_ALGO`** so you can pick a mode without recompiling:
///
/// - `auto` (default): `SearchParams::new()` + `itopk` only — same style as `maxivf_cagra`
///   assignment in this repo.
/// - `single` / `single_cta`: `SearchAlgo::SINGLE_CTA` + `itopk` only.
/// - `multi` / `multi_cta`: `SearchAlgo::MULTI_CTA` + `itopk` only.
/// - `multi_tuned`: `MULTI_CTA` with `team_size=8` and `thread_block_size=256` (historical
///   workaround attempt; your recent logs showed this path still illegal-accessing on SBSA).
#[derive(Debug, Clone, Copy)]
enum CentroidSearchAlgoMode {
    Auto,
    SingleCta,
    MultiCta,
    MultiCtaTuned,
}

fn centroid_search_algo_from_env() -> CentroidSearchAlgoMode {
    match std::env::var("FASTPLAID_CAGRA_CENTROID_SEARCH_ALGO") {
        Ok(s) => match s.to_ascii_lowercase().as_str() {
            "" | "auto" => CentroidSearchAlgoMode::Auto,
            "single" | "single_cta" => CentroidSearchAlgoMode::SingleCta,
            "multi" | "multi_cta" => CentroidSearchAlgoMode::MultiCta,
            "multi_tuned" => CentroidSearchAlgoMode::MultiCtaTuned,
            other => {
                eprintln!(
                    "[fast-plaid][cagra-centroids] FASTPLAID_CAGRA_CENTROID_SEARCH_ALGO={other:?} \
                     is not one of auto|single|multi|multi_tuned; using auto"
                );
                CentroidSearchAlgoMode::Auto
            }
        },
        Err(_) => CentroidSearchAlgoMode::Auto,
    }
}

/// cuVS CAGRA search has reproduced illegal global memory access on SBSA when `n_queries` is
/// tiny (e.g. a few dozen IVF probe tokens) while maxIVF training uses huge batches on the same
/// index size. Padding by **repeating the whole query block** along batch dim preserves ANN
/// results for rows `0..n_queries` (row `i` and `i + n_queries` are identical queries).
fn min_padded_cagra_query_rows() -> usize {
    match std::env::var("FASTPLAID_CAGRA_CENTROID_QUERY_PAD_TO") {
        Ok(s) => match s.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => 256,
        },
        Err(_) => 256,
    }
}

/// GPU-backed CAGRA state built over IVF centroid rows.
pub struct HnswCentroidState {
    index: Mutex<Index>,
    resources: Resources,
    itopk_size: usize,
    dim: usize,
    n: usize,
    build_id: u64,
    topk_calls: AtomicU64,
    /// Retained host dataset buffer. cuvs-rs's `ManagedTensor::from(&Array2)` builds a
    /// non-owning DLPack view (dlpack.rs:96), and at search time `cuvsCagraSearch` reaches
    /// back through the dataset pointer in `compute_distance.hpp`. Dropping the host buffer
    /// before search yields an illegal global access on the first query (reproduced on FIQA
    /// at n_centroids=192381). Pin the buffer for the lifetime of the Index.
    _host_dataset: Array2<f32>,
}

impl HnswCentroidState {
    pub(super) fn build_centroids(
        centroids: &Tensor,
        params: &super::HnswBuildParams,
    ) -> Result<Arc<Self>> {
        static BUILD_SEQ: AtomicU64 = AtomicU64::new(1);

        // cuVS ManagedTensor currently hard-codes device_id=0 when copying host->device.
        // For correctness, reject building/searching on nonzero CUDA devices.
        if let Device::Cuda(idx) = centroids.device() {
            anyhow::ensure!(
                idx == 0,
                "centroid_index='cagra' currently requires CUDA device 0, got cuda:{}",
                idx
            );
        }

        // Move centroids to CPU f32 and copy into an ndarray matrix for cuVS.
        let cpu = centroids
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();
        let sz = cpu.size();
        anyhow::ensure!(
            sz.len() == 2,
            "expected centroids matrix [num_centroids, dim], got shape {:?}",
            sz
        );
        let n = usize::try_from(sz[0]).map_err(|_| anyhow!("centroid count out of range"))?;
        let dim = usize::try_from(sz[1]).map_err(|_| anyhow!("embedding dim out of range"))?;
        anyhow::ensure!(n > 0 && dim > 0, "empty centroids");

        ensure_tensor_all_finite(&cpu, "centroid CAGRA index build (centroid matrix)")?;

        let n_el = n
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("centroid matrix element count overflow"))?;
        let mut row_major = vec![0f32; n_el];
        cpu.copy_data(&mut row_major, n_el);

        let dataset =
            Array2::from_shape_vec((n, dim), row_major).map_err(|e| anyhow!("{e}"))?;

        let mut index_params = IndexParams::new().context("cuvs IndexParams::new failed")?;
        index_params = index_params.set_graph_degree(params.m as usize);
        index_params =
            index_params.set_intermediate_graph_degree(params.ef_construction as usize);
        cagra_index_params_set_cosine_expanded(&mut index_params);

        let resources = Resources::new().context("cuvs Resources::new failed")?;

        let build_id = BUILD_SEQ.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{build_id}] building centroid CAGRA index: metric=CosineExpanded n={n} dim={dim} graph_degree={} intermediate_graph_degree={} itopk_size={}",
            params.m,
            params.ef_construction,
            params.ef_search
        );
        let index = Index::build(&resources, &index_params, &dataset)
            .context("cuvs CAGRA Index::build failed")?;
        eprintln!("[fast-plaid][cagra-centroids][build#{build_id}] build done");

        let itopk_size = usize::try_from(params.ef_search).unwrap_or(256).max(1);

        Ok(Arc::new(Self {
            index: Mutex::new(index),
            resources,
            itopk_size,
            dim,
            n,
            build_id,
            topk_calls: AtomicU64::new(0),
            _host_dataset: dataset,
        }))
    }

    pub(super) fn topk_centroids(
        &self,
        centroids_for_scores: &Tensor,
        queries: &Tensor,
        k: i64,
    ) -> Result<(Tensor, Tensor)> {
        let call_n = self.topk_calls.fetch_add(1, Ordering::Relaxed) + 1;

        if let Device::Cuda(idx) = queries.device() {
            anyhow::ensure!(
                idx == 0,
                "centroid_index='cagra' currently requires CUDA device 0, got cuda:{}",
                idx
            );
        }

        let [n_q_tokens, dim] = queries.size()[..] else {
            anyhow::bail!("bad query tensor shape {:?}", queries.size())
        };
        let dim_usize =
            usize::try_from(dim).map_err(|_| anyhow!("embedding dim {:?}", dim))?;
        anyhow::ensure!(
            dim_usize == self.dim,
            "CAGRA centroid index dim mismatch: index dim {} vs query dim {}",
            self.dim,
            dim_usize
        );

        let n_centroids = centroids_for_scores.size()[0];
        anyhow::ensure!(
            usize::try_from(n_centroids).unwrap_or(0) == self.n,
            "CAGRA centroid index size mismatch: index n={} vs centroid tensor n={}",
            self.n,
            n_centroids
        );
        let kd = k.min(n_centroids).max(1);
        let k_usize = usize::try_from(kd)?;
        let n_q_us = usize::try_from(n_q_tokens)?;
        let algo_mode = centroid_search_algo_from_env();

        ensure_tensor_all_finite(
            queries,
            "centroid CAGRA search (query token embeddings)",
        )?;

        let tch_err = |stage: &str, e: TchError| anyhow!("tch error at {stage}: {e}");

        let pad_to = min_padded_cagra_query_rows();
        let repeats: i64 = if n_q_us < pad_to {
            (((pad_to + n_q_us - 1) / n_q_us) as i64).max(1)
        } else {
            1
        };
        let n_q_search = n_q_us
            .checked_mul(repeats as usize)
            .ok_or_else(|| anyhow!("padded query count overflow"))?;
        let queries_search = if repeats > 1 {
            if call_n <= 3 || call_n % 50 == 0 {
                eprintln!(
                    "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] padding queries: n_q={} repeat_along_batch={} padded_n_q={} (FASTPLAID_CAGRA_CENTROID_QUERY_PAD_TO={})",
                    self.build_id,
                    n_q_us,
                    repeats,
                    n_q_search,
                    pad_to
                );
            }
            queries.repeat(&[repeats, 1]).contiguous()
        } else {
            queries.shallow_clone()
        };

        if call_n <= 3 || call_n % 50 == 0 {
            eprintln!(
                "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] q_tokens={} dim={} k={} itopk_size={} search_algo_mode={algo_mode:?} (override with FASTPLAID_CAGRA_CENTROID_SEARCH_ALGO)",
                self.build_id,
                n_q_tokens,
                dim_usize,
                kd,
                k_usize.max(self.itopk_size)
            );
        }

        // Copy queries to host f32 ndarray, then to device via ManagedTensor.
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=query_to_cpu",
            self.build_id
        );
        let q_cpu = queries_search
            .f_to_device(Device::Cpu)
            .map_err(|e| tch_err("queries.f_to_device(Cpu)", e))?
            .f_to_kind(Kind::Float)
            .map_err(|e| tch_err("queries_cpu.f_to_kind(Float)", e))?
            .contiguous();
        let n_el_q = n_q_search
            .checked_mul(dim_usize)
            .ok_or_else(|| anyhow!("query buffer size overflow"))?;
        let mut q_row_major = vec![0f32; n_el_q];
        q_cpu.copy_data(&mut q_row_major, n_el_q);
        let q_host =
            Array2::from_shape_vec((n_q_search, dim_usize), q_row_major).map_err(|e| anyhow!("{e}"))?;

        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=queries_to_device",
            self.build_id
        );
        let queries_dev = ManagedTensor::from(&q_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (queries) failed")?;

        // Allocate device outputs.
        let mut neighbors_host = Array2::<u32>::zeros((n_q_search, k_usize));
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=alloc_outputs_to_device",
            self.build_id
        );
        let neighbors_dev = ManagedTensor::from(&neighbors_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (neighbors) failed")?;

        let distances_host = Array2::<f32>::zeros((n_q_search, k_usize));
        let distances_dev = ManagedTensor::from(&distances_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (distances) failed")?;

        let mut search_params =
            SearchParams::new().context("cuvs SearchParams::new failed")?;
        let itopk = k_usize.max(self.itopk_size);
        search_params = search_params.set_itopk_size(itopk);
        match algo_mode {
            CentroidSearchAlgoMode::Auto => {}
            CentroidSearchAlgoMode::SingleCta => {
                search_params = search_params.set_algo(SearchAlgo::SINGLE_CTA);
            }
            CentroidSearchAlgoMode::MultiCta => {
                search_params = search_params.set_algo(SearchAlgo::MULTI_CTA);
            }
            CentroidSearchAlgoMode::MultiCtaTuned => {
                search_params = search_params
                    .set_algo(SearchAlgo::MULTI_CTA)
                    .set_team_size(8)
                    .set_thread_block_size(256);
            }
        }

        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=cuvs_search",
            self.build_id
        );
        self.index
            .lock()
            .search(
                &self.resources,
                &search_params,
                &queries_dev,
                &neighbors_dev,
                &distances_dev,
            )
            .context("cuvs CAGRA Index::search failed")?;

        // cuVS operations are asynchronous on the cuVS-managed stream. Sync here so that
        // any CUDA failures (e.g. illegal memory access) are attributed to the search call
        // rather than surfacing later during an unrelated Torch op.
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=sync_after_search",
            self.build_id
        );
        self.resources
            .sync_stream()
            .context("cuvs Resources::sync_stream after CAGRA search failed")?;

        // Copy neighbor ids back to host.
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=neighbors_to_host",
            self.build_id
        );
        neighbors_dev
            .to_host(&self.resources, &mut neighbors_host)
            .context("cuvs ManagedTensor::to_host (neighbors) failed")?;

        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=sync_after_neighbors_to_host",
            self.build_id
        );
        self.resources
            .sync_stream()
            .context("cuvs Resources::sync_stream after neighbors to_host failed")?;

        // Convert neighbor IDs to a tensor on the original query device.
        eprintln!(
            "[fast-plaid][cagra-centroids][build#{}][topk#{call_n}] stage=neighbors_to_torch",
            self.build_id
        );
        let mut flat_ids: Vec<i64> = Vec::with_capacity(n_q_us * k_usize);
        for i in 0..n_q_us {
            for j in 0..k_usize {
                flat_ids.push(neighbors_host[(i, j)] as i64);
            }
        }

        let ids_on_q = Tensor::from_slice(&flat_ids)
            .view([n_q_tokens, kd])
            .f_to_device(queries.device())
            .map_err(|e| tch_err("ids.f_to_device(query_device)", e))?
            .f_to_kind(Kind::Int64)
            .map_err(|e| tch_err("ids_on_q.f_to_kind(Int64)", e))?
            .contiguous();

        // Recompute exact dot-product scores on the centroid tensor.
        let gathered = centroids_for_scores
            .index_select(0, &ids_on_q.flatten(0, -1))
            .view([n_q_tokens, kd, -1])
            .to_kind(queries.kind());

        let q_shaped = queries.unsqueeze(1);
        let dots = gathered
            .mul(&q_shaped)
            .sum_dim_intlist(&[-1i64][..], false, None::<Kind>);

        Ok((ids_on_q, dots.contiguous()))
    }
}
