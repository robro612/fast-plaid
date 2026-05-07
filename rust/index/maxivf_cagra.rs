use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tch::{Device, Kind, Tensor};
use tch::IndexOp;

#[cfg(feature = "cagra")]
use anyhow::Context;
#[cfg(feature = "cagra")]
use cuvs::cagra::{Index, IndexParams, SearchAlgo, SearchParams};
#[cfg(feature = "cagra")]
use cuvs::{ManagedTensor, Resources};
#[cfg(feature = "cagra")]
use ndarray::Array2;
#[cfg(feature = "cagra")]
use crate::utils::cagra_metric::cagra_index_params_set_cosine_expanded;
#[cfg(feature = "cagra")]
use crate::utils::tensor_finite::ensure_tensor_all_finite;
#[cfg(feature = "cagra")]
use std::cell::Cell;

#[cfg(feature = "cagra")]
fn cuda_synchronize_device(device: Device) {
    if let Device::Cuda(i) = device {
        tch::Cuda::synchronize(i as i64);
    }
}

#[cfg(feature = "cagra")]
fn maxivf_query_pad_to_min_rows() -> usize {
    match std::env::var("FASTPLAID_CAGRA_MAXIVF_QUERY_PAD_TO") {
        Ok(s) => match s.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => 256,
        },
        Err(_) => 256,
    }
}

/// Repeat queries along batch so `n_queries >= pad_to` when the batch is tiny (libcuvs CAGRA has
/// reproduced illegal global access on SBSA for small `n_queries` at large index sizes).
#[cfg(feature = "cagra")]
fn pad_token_batch_along_rows(batch: Tensor, pad_to: usize) -> (Tensor, usize, usize) {
    let orig = batch.size()[0] as usize;
    if orig == 0 {
        return (batch, 0, 0);
    }
    if orig >= pad_to {
        return (batch, orig, orig);
    }
    let repeats = (((pad_to + orig - 1) / orig) as i64).max(1);
    let padded = batch.repeat(&[repeats, 1]).contiguous();
    let search_n = padded.size()[0] as usize;
    (padded, orig, search_n)
}

#[cfg(feature = "cagra")]
fn maxivf_search_params_with_env(itopk: usize) -> Result<SearchParams> {
    let mut sp = SearchParams::new().context("cuvs SearchParams::new failed")?;
    sp = sp.set_itopk_size(itopk.max(1));
    match std::env::var("FASTPLAID_CAGRA_MAXIVF_SEARCH_ALGO") {
        Ok(s) => match s.to_ascii_lowercase().as_str() {
            "" | "auto" => {}
            "single" | "single_cta" => {
                sp = sp.set_algo(SearchAlgo::SINGLE_CTA);
            }
            "multi" | "multi_cta" => {
                sp = sp.set_algo(SearchAlgo::MULTI_CTA);
            }
            "multi_tuned" => {
                sp = sp
                    .set_algo(SearchAlgo::MULTI_CTA)
                    .set_team_size(8)
                    .set_thread_block_size(256);
            }
            other => {
                eprintln!(
                    "[fast-plaid][maxIVF_cagra] FASTPLAID_CAGRA_MAXIVF_SEARCH_ALGO={other:?} is not one of \
                     auto|single|multi|multi_tuned; using auto"
                );
            }
        },
        Err(_) => {}
    }
    Ok(sp)
}

/// Resolved policy for empty clusters detected at the end of an outer M-step.
#[cfg(feature = "cagra")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmptyHandling {
    /// Keep the previous-iteration anchor for empty clusters (the historic default;
    /// produces duplicate anchor rows but never shrinks the anchor set).
    Sticky,
    /// Replace each empty anchor with a fresh randomly-sampled token (uniform-by-token).
    Reseed,
    /// Drop empty clusters entirely so subsequent iterations work on a smaller anchor set.
    /// Avoids degenerate duplicates polluting the next CAGRA build and the final centroid
    /// index, at the cost of a shrinking n_anchors per iter.
    Prune,
}

#[cfg(feature = "cagra")]
fn parse_empty_handling(s: &str) -> Result<EmptyHandling> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "sticky" => EmptyHandling::Sticky,
        "reseed" => EmptyHandling::Reseed,
        "prune" => EmptyHandling::Prune,
        other => {
            return Err(anyhow!(
                "unknown empty_handling {other:?}; expected 'sticky' | 'reseed' | 'prune'"
            ))
        }
    })
}

fn l2_normalize_rows(x: &Tensor, eps: f64) -> Tensor {
    let norms = x
        .to_kind(Kind::Float)
        .norm_scalaropt_dim(2.0, &[1], true)
        .clamp_min(eps);
    (x.to_kind(Kind::Float) / norms).to_kind(x.kind())
}

fn total_tokens(embeddings: &[PyTensor]) -> Result<i64> {
    let mut tot: i64 = 0;
    for t in embeddings {
        let sz = t.size();
        if sz.len() != 2 {
            return Err(anyhow!("expected 2D token tensor, got shape {:?}", sz));
        }
        tot += sz[0];
    }
    Ok(tot)
}

fn default_n_anchors(n_tokens_total: i64) -> usize {
    // linear rule: 0.025 * n_tokens_total
    let n = ((n_tokens_total as f64) * 0.025).floor() as i64;
    n.max(1) as usize
}

/// Doc-length CDF for uniform-random token sampling (weighted by tokens per doc).
fn token_sampling_plan(embeddings: &[PyTensor]) -> Result<(Vec<i64>, Vec<i64>, i64, i64)> {
    let dim = embeddings
        .first()
        .ok_or_else(|| anyhow!("empty embeddings"))?
        .size()[1];
    let mut lens: Vec<i64> = Vec::with_capacity(embeddings.len());
    let mut cum: Vec<i64> = Vec::with_capacity(embeddings.len());
    let mut running: i64 = 0;
    for t in embeddings {
        let l = t.size()[0];
        lens.push(l);
        running += l;
        cum.push(running);
    }
    if running <= 0 {
        return Err(anyhow!("no tokens available for anchor sampling"));
    }
    Ok((lens, cum, running, dim))
}

/// Sample `n_rows` token rows uniformly at random over all tokens (same scheme as init).
fn sample_token_rows_weighted(
    embeddings: &[PyTensor],
    n_rows: usize,
    device: Device,
    rng: &mut StdRng,
    lens: &[i64],
    cum: &[i64],
    running: i64,
    dim: i64,
) -> Result<Tensor> {
    let mut rows: Vec<Tensor> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let r = rng.random_range(0..running);
        let doc_idx = match cum.binary_search(&r) {
            Ok(i) => i,
            Err(i) => i,
        };
        let doc_idx = doc_idx.min(embeddings.len() - 1);
        let doc_len = lens[doc_idx];
        let tok = if doc_len > 0 {
            rng.random_range(0..doc_len)
        } else {
            0
        };
        let row = embeddings[doc_idx]
            .shallow_clone()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .i(tok);
        rows.push(row);
    }
    Ok(
        Tensor::stack(&rows, 0)
            .to_device(device)
            .to_kind(Kind::Float)
            .view([n_rows as i64, dim]),
    )
}

#[cfg(not(feature = "cagra"))]
pub fn maxivf_cagra_anchors_impl(
    _embeddings: Vec<PyTensor>,
    _device: Device,
    _n_anchors: Option<usize>,
    _n_iter: usize,
    _graph_degree: usize,
    _intermediate_graph_degree: usize,
    _itopk_size: usize,
    _token_batch_size: usize,
    _seed: Option<u64>,
    _normalize_anchors: bool,
    _empty_handling: Option<String>,
) -> Result<Tensor> {
    Err(anyhow!(
        "anchor_method='maxIVF_cagra' requires building fast_plaid_rust with Cargo feature `cagra` \
         (cuVS). Rebuild with e.g. `maturin develop --features cagra`."
    ))
}

#[cfg(feature = "cagra")]
pub fn maxivf_cagra_anchors_impl(
    embeddings: Vec<PyTensor>,
    device: Device,
    n_anchors: Option<usize>,
    n_iter: usize,
    graph_degree: usize,
    intermediate_graph_degree: usize,
    itopk_size: usize,
    token_batch_size: usize,
    seed: Option<u64>,
    normalize_anchors: bool,
    empty_handling: Option<String>,
) -> Result<Tensor> {
    let n_tokens_total = total_tokens(&embeddings)?;
    let n_anchors = n_anchors.unwrap_or_else(|| default_n_anchors(n_tokens_total));
    let empty_handling = match empty_handling.as_deref() {
        Some(s) => parse_empty_handling(s)?,
        None => EmptyHandling::Sticky,
    };

    eprintln!(
        "[fast-plaid][maxIVF_cagra] start: n_docs={} n_tokens_total={} n_anchors={} n_iter={} dim?=unknown graph_degree={} intermediate_graph_degree={} itopk_size={} token_batch_size={} normalize_anchors={} empty_handling={:?} device={:?}",
        embeddings.len(),
        n_tokens_total,
        n_anchors,
        n_iter,
        graph_degree,
        intermediate_graph_degree,
        itopk_size,
        token_batch_size,
        normalize_anchors,
        empty_handling,
        device
    );

    let (lens, cum, running, dim_i64) =
        token_sampling_plan(&embeddings).context("token sampling plan")?;
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
    };
    let mut anchors = sample_token_rows_weighted(
        &embeddings,
        n_anchors,
        device,
        &mut rng,
        &lens,
        &cum,
        running,
        dim_i64,
    )
    .context("failed to random-sample initial anchors")?;

    // When training with L2-normalized anchors, the first outer iteration must use the same
    // geometry as later ones: CAGRA is built with CosineExpanded. If we only normalize *after*
    // the first M-step, iteration 1 searches an index over raw token norms while iteration 2
    // (and libcuvs search kernels) sees unit-norm rows — a mismatch that has reproduced
    // illegal memory access on SBSA libcuvs on the first search of iteration 2.
    if normalize_anchors {
        anchors = l2_normalize_rows(&anchors, 1e-12).to_kind(Kind::Float);
        eprintln!(
            "[fast-plaid][maxIVF_cagra] initial anchors L2-normalized (match post-iter geometry)"
        );
    }

    let dim = anchors.size()[1] as usize;

    for iter in 0..n_iter {
        let res = Resources::new().context("cuvs Resources::new failed")?;
        let anchors_at_iter_start = anchors.shallow_clone();
        // n_anchors_cur = current anchor count for this iter. Equals the initial
        // n_anchors except in `prune` mode where it shrinks monotonically.
        let n_anchors_cur = anchors.size()[0] as usize;
        eprintln!(
            "[fast-plaid][maxIVF_cagra][iter {}/{}] build CAGRA over anchors: n={} dim={}",
            iter + 1,
            n_iter,
            n_anchors_cur,
            dim
        );
        // Build CAGRA graph over current anchors (on host for build API).
        let anchors_cpu = anchors.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        ensure_tensor_all_finite(
            &anchors_cpu,
            "maxIVF_cagra CAGRA graph build (anchor matrix)",
        )?;
        let n = anchors_cpu.size()[0] as usize;
        let mut buf = vec![0f32; n * dim];
        anchors_cpu.copy_data(&mut buf, n * dim);
        let dataset = Array2::from_shape_vec((n, dim), buf).map_err(|e| anyhow!("{e}"))?;

        let mut ip = IndexParams::new().context("cuvs IndexParams::new failed")?;
        ip = ip
            .set_graph_degree(graph_degree)
            .set_intermediate_graph_degree(intermediate_graph_degree);
        cagra_index_params_set_cosine_expanded(&mut ip);
        let index = Index::build(&res, &ip, &dataset).context("cuvs CAGRA Index::build failed")?;
        eprintln!("[fast-plaid][maxIVF_cagra][iter {}/{}] CAGRA build done", iter + 1, n_iter);

        // Accumulators on device (float32). Sized to the **current** anchor count so prune
        // mode's shrinking n_anchors doesn't corrupt index_add_ targets.
        let mut sums = Tensor::zeros(&[n_anchors_cur as i64, anchors.size()[1]], (Kind::Float, device));
        let mut counts = Tensor::zeros(&[n_anchors_cur as i64], (Kind::Float, device));

        let sp = maxivf_search_params_with_env(itopk_size)?;
        let query_pad_to = maxivf_query_pad_to_min_rows();

        let checked_first_token_batch = Cell::new(false);
        // Iterate tokens in batches.
        let mut batch_rows: Vec<Tensor> = Vec::new();
        let mut batch_count: i64 = 0;
        let mut flushed_batches: u64 = 0;

        let mut flush_batch = |batch_rows: &mut Vec<Tensor>, batch_count: &mut i64| -> Result<()> {
            if *batch_count == 0 {
                return Ok(());
            }
            let batch = Tensor::cat(batch_rows, 0)
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .contiguous();
            if !checked_first_token_batch.get() {
                ensure_tensor_all_finite(
                    &batch,
                    "maxIVF_cagra CAGRA search (first token batch of iteration)",
                )?;
                checked_first_token_batch.set(true);
            }
            let (batch_search, orig_bsz, search_bsz) = pad_token_batch_along_rows(batch, query_pad_to);
            if orig_bsz == 0 {
                batch_rows.clear();
                *batch_count = 0;
                return Ok(());
            }
            let mut qbuf = vec![0f32; search_bsz * dim];
            batch_search.copy_data(&mut qbuf, search_bsz * dim);
            let q_host =
                Array2::from_shape_vec((search_bsz, dim), qbuf).map_err(|e| anyhow!("{e}"))?;

            let queries_dev = ManagedTensor::from(&q_host).to_device(&res)?;
            let mut neighbors_host = Array2::<u32>::zeros((search_bsz, 1));
            let neighbors_dev = ManagedTensor::from(&neighbors_host).to_device(&res)?;
            let distances_host = Array2::<f32>::zeros((search_bsz, 1));
            let distances_dev = ManagedTensor::from(&distances_host).to_device(&res)?;

            index.search(&res, &sp, &queries_dev, &neighbors_dev, &distances_dev)?;
            neighbors_dev.to_host(&res, &mut neighbors_host)?;
            cuda_synchronize_device(device);

            // Guard: OOB labels poison GPU index_add_ and surface as illegal access later.
            let mut ids: Vec<i64> = Vec::with_capacity(orig_bsz);
            for i in 0..orig_bsz {
                let nid = neighbors_host[(i, 0)] as u64;
                let nmax = n_anchors_cur as u64;
                if nid >= nmax {
                    return Err(anyhow!(
                        "maxIVF_cagra: CAGRA neighbor id {nid} out of range [0, {n_anchors_cur}) \
                         (query row {i}, padded_search_bsz={search_bsz})"
                    ));
                }
                ids.push(nid as i64);
            }
            let ids_t = Tensor::from_slice(&ids).to_device(device).to_kind(Kind::Int64);
            let ones = Tensor::ones(&[orig_bsz as i64], (Kind::Float, device));
            let _ = counts.index_add_(0, &ids_t, &ones);

            let batch_orig = batch_search.narrow(0, 0, orig_bsz as i64);
            let batch_dev = batch_orig.to_device(device);
            let _ = sums.index_add_(0, &ids_t, &batch_dev);

            batch_rows.clear();
            *batch_count = 0;
            Ok(())
        };

        for doc in &embeddings {
            let t = doc.shallow_clone();
            let sz = t.size();
            let rows = sz[0];
            if rows == 0 {
                continue;
            }
            // Chunk doc tokens into batches.
            let mut start: i64 = 0;
            while start < rows {
                let take = (rows - start).min((token_batch_size as i64).max(1));
                let chunk = t.narrow(0, start, take);
                batch_rows.push(chunk);
                batch_count += take;
                if batch_count >= token_batch_size as i64 {
                    flush_batch(&mut batch_rows, &mut batch_count)?;
                    flushed_batches += 1;
                    if flushed_batches <= 3 || flushed_batches % 25 == 0 {
                        eprintln!(
                            "[fast-plaid][maxIVF_cagra][iter {}/{}] flushed batch #{flushed_batches} (token_batch_size={})",
                            iter + 1,
                            n_iter,
                            token_batch_size
                        );
                    }
                }
                start += take;
            }
        }
        flush_batch(&mut batch_rows, &mut batch_count)?;

        // M-step: means for assigned clusters. Empty clusters (count==0) need explicit
        // handling — all-zero rows from `sums / clamp_min(count)` degenerate CosineExpanded
        // CAGRA on the next outer iteration. Three policies (selected via `empty_handling`):
        //
        // - Sticky: keep `anchors_at_iter_start[empty_idx]` (historic default; produces
        //   duplicate rows but never shrinks the anchor set).
        // - Reseed: replace each empty anchor with a fresh randomly-sampled token.
        // - Prune:  drop empty rows entirely so subsequent iterations work on a smaller
        //   anchor set. Avoids polluting the next CAGRA build with degenerate duplicates;
        //   `n_anchors` shrinks monotonically per iter (and is reflected in the returned
        //   tensor's leading dimension).
        let counts_flat = counts.view([-1]);
        let mask_f = counts_flat.gt(0.0).unsqueeze(1).to_kind(Kind::Float);
        let ones_m = Tensor::ones_like(&mask_f);
        let denom = counts_flat.unsqueeze(1).clamp_min(1.0);
        let cluster_means = (sums / denom).to_kind(Kind::Float);
        anchors = cluster_means * &mask_f + &anchors_at_iter_start * (&ones_m - &mask_f);

        let nz = counts_flat.eq(0.0).nonzero();
        let n_empty = nz.size()[0] as usize;
        if n_empty > 0 {
            match empty_handling {
                EmptyHandling::Sticky => {
                    eprintln!(
                        "[fast-plaid][maxIVF_cagra][iter {}/{}] {n_empty} empty anchors — kept previous values (sticky)",
                        iter + 1,
                        n_iter,
                    );
                }
                EmptyHandling::Reseed => {
                    let empty_idx = nz.select(1, 0).to_kind(Kind::Int64).to_device(device);
                    let reseed = sample_token_rows_weighted(
                        &embeddings,
                        n_empty,
                        device,
                        &mut rng,
                        &lens,
                        &cum,
                        running,
                        dim_i64,
                    )
                    .context("re-seed empty anchors (token sample)")?;
                    anchors = anchors.index_copy(0, &empty_idx, &reseed);
                    eprintln!(
                        "[fast-plaid][maxIVF_cagra][iter {}/{}] re-seeded {n_empty} empty anchors with random tokens",
                        iter + 1,
                        n_iter,
                    );
                }
                EmptyHandling::Prune => {
                    // Build a keep_mask = (count > 0) and gather the surviving rows.
                    let keep_idx = counts_flat.gt(0.0).nonzero().select(1, 0).to_kind(Kind::Int64).to_device(device);
                    let n_kept = keep_idx.size()[0] as usize;
                    if n_kept == 0 {
                        return Err(anyhow!(
                            "maxIVF_cagra prune: every cluster was empty at iter {}/{} — refusing to drop all anchors",
                            iter + 1,
                            n_iter
                        ));
                    }
                    anchors = anchors.index_select(0, &keep_idx).contiguous();
                    eprintln!(
                        "[fast-plaid][maxIVF_cagra][iter {}/{}] pruned {n_empty} empty anchors; n_anchors {} -> {}",
                        iter + 1,
                        n_iter,
                        n_anchors_cur,
                        n_kept,
                    );
                }
            }
        }
        if normalize_anchors {
            anchors = l2_normalize_rows(&anchors, 1e-12).to_kind(Kind::Float);
        }
        let anchors_post = anchors.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        ensure_tensor_all_finite(
            &anchors_post,
            "maxIVF_cagra anchors after iteration (pre next build)",
        )?;
        eprintln!(
            "[fast-plaid][maxIVF_cagra][iter {}/{}] recompute done (normalize_anchors={})",
            iter + 1,
            n_iter,
            normalize_anchors
        );
    }

    // Return float16 anchors (consistent with centroids usage elsewhere).
    eprintln!("[fast-plaid][maxIVF_cagra] done");
    Ok(anchors.to_kind(Kind::Half))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn maxivf_cagra_anchors(
    _py: Python<'_>,
    embeddings: Vec<PyTensor>,
    device: String,
    n_anchors: Option<usize>,
    n_iter: usize,
    graph_degree: usize,
    intermediate_graph_degree: usize,
    itopk_size: usize,
    token_batch_size: usize,
    seed: Option<u64>,
    normalize_anchors: bool,
    empty_handling: Option<String>,
) -> PyResult<PyTensor> {
    let device = crate::search::load::get_device(&device)?;
    let out = maxivf_cagra_anchors_impl(
        embeddings,
        device,
        n_anchors,
        n_iter,
        graph_degree,
        intermediate_graph_degree,
        itopk_size,
        token_batch_size,
        seed,
        normalize_anchors,
        empty_handling,
    )
    .map_err(crate::utils::errors::anyhow_to_pyerr)?;
    Ok(PyTensor(out))
}

