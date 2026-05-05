//! Faiss HNSW index over IVF centroids (CPU; via faiss-next).
//!
//! Neighbor IDs come from approximate L2 graph search among centroids (appropriate for
//! L2-normalized ColBERT-style embeddings). Returned dot-product scores are recomputed on
//! the centroid tensor.
#![cfg(feature = "hnsw")]

use std::ops::Mul;

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use faiss_next::{Index, IndexBuilder, ParameterSpace};
use parking_lot::Mutex;
use tch::{Device, Kind, Tensor};

/// CPU Faiss HNSW state built over IVF centroid rows.
/// CPU-bound Faiss HNSW index over centroid rows (see module docs).
pub struct HnswCentroidState {
    index: Mutex<faiss_next::IndexImpl>,
}

impl HnswCentroidState {
    pub(super) fn build_centroids(
        centroids: &Tensor,
        params: &super::HnswBuildParams,
    ) -> Result<Arc<Self>> {
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

        let n_el = n
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("centroid matrix element count overflow"))?;
        let mut row_major = vec![0f32; n_el];
        cpu.copy_data(&mut row_major, n_el);

        let dim_u32 = u32::try_from(dim).map_err(|_| anyhow!("dim {} exceeds u32", dim))?;
        let m = usize::try_from(params.m).map_err(|_| anyhow!("HNSW M out of range"))?;

        let mut index = IndexBuilder::new(dim_u32)
            .hnsw(m)
            .l2()
            .build()
            .context("faiss IndexBuilder HNSW build failed (is libfaiss installed?)")?;

        let ps = ParameterSpace::new().context("faiss ParameterSpace::new failed")?;
        let pstr = format!(
            "efConstruction={},efSearch={}",
            params.ef_construction, params.ef_search
        );
        ps.set_index_parameters(&mut index, &pstr)
            .with_context(|| format!("faiss set_index_parameters({pstr}) failed"))?;

        index
            .add(&row_major)
            .context("faiss HNSW Index::add (centroids) failed")?;

        anyhow::ensure!(
            index.ntotal() == n as u64,
            "faiss index ntotal {} != centroid count {}",
            index.ntotal(),
            n
        );

        Ok(Arc::new(Self {
            index: Mutex::new(index),
        }))
    }

    pub(super) fn topk_centroids(
        &self,
        centroids_for_scores: &Tensor,
        queries: &Tensor,
        k: i64,
    ) -> Result<(Tensor, Tensor)> {
        let [n_q_tokens, dim] = queries.size()[..] else {
            anyhow::bail!("bad query tensor shape {:?}", queries.size())
        };
        let dim = usize::try_from(dim).map_err(|_| anyhow!("embedding dim {:?}", dim))?;
        let n_centroids = centroids_for_scores.size()[0];

        let kd = k.min(n_centroids).max(1);
        let k_usize = usize::try_from(kd)?;
        let n_q_us = usize::try_from(n_q_tokens)?;

        let q_cpu = queries
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();
        let n_el_q = n_q_us
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("query buffer size overflow"))?;
        let mut q_row_major = vec![0f32; n_el_q];
        q_cpu.copy_data(&mut q_row_major, n_el_q);

        let search_result = self
            .index
            .lock()
            .search(&q_row_major, k_usize)
            .context("faiss HNSW Index::search failed")?;

        let mut flat_ids: Vec<i64> = Vec::with_capacity(n_q_us * k_usize);
        for lab in &search_result.labels {
            let id = if lab.is_none() {
                -1i64
            } else {
                lab.as_repr()
            };
            flat_ids.push(id);
        }
        debug_assert_eq!(flat_ids.len(), n_q_us * k_usize);

        let ids_on_q = Tensor::from_slice(&flat_ids)
            .view([n_q_tokens, kd])
            .to_device(queries.device())
            .to_kind(Kind::Int64)
            .contiguous();

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
