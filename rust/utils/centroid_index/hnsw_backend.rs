//! CAGRA index over IVF centroids (GPU; via cuVS).
//!
//! Neighbor IDs come from an approximate graph search among centroids using the
//! cuVS CAGRA index. Returned dot-product scores are recomputed on the centroid
//! tensor to stay consistent with the dense backend.
#![cfg(feature = "cagra")]

use std::sync::Arc;
use std::ops::Mul;

use anyhow::{anyhow, Context, Result};
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources};
use ndarray::Array2;
use parking_lot::Mutex;
use tch::{Device, Kind, Tensor};

/// GPU-backed CAGRA state built over IVF centroid rows.
pub struct HnswCentroidState {
    index: Mutex<Index>,
    resources: Resources,
    itopk_size: usize,
}

impl HnswCentroidState {
    pub(super) fn build_centroids(
        centroids: &Tensor,
        params: &super::HnswBuildParams,
    ) -> Result<Arc<Self>> {
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

        let resources = Resources::new().context("cuvs Resources::new failed")?;
        let index = Index::build(&resources, &index_params, &dataset)
            .context("cuvs CAGRA Index::build failed")?;

        Ok(Arc::new(Self {
            index: Mutex::new(index),
            resources,
            itopk_size: usize::try_from(params.ef_search)
                .unwrap_or(64)
                .max(1),
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
        let dim_usize =
            usize::try_from(dim).map_err(|_| anyhow!("embedding dim {:?}", dim))?;

        let n_centroids = centroids_for_scores.size()[0];
        let kd = k.min(n_centroids).max(1);
        let k_usize = usize::try_from(kd)?;
        let n_q_us = usize::try_from(n_q_tokens)?;

        // Copy queries to host f32 ndarray, then to device via ManagedTensor.
        let q_cpu = queries
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();
        let n_el_q = n_q_us
            .checked_mul(dim_usize)
            .ok_or_else(|| anyhow!("query buffer size overflow"))?;
        let mut q_row_major = vec![0f32; n_el_q];
        q_cpu.copy_data(&mut q_row_major, n_el_q);
        let q_host =
            Array2::from_shape_vec((n_q_us, dim_usize), q_row_major).map_err(|e| anyhow!("{e}"))?;

        let queries_dev = ManagedTensor::from(&q_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (queries) failed")?;

        // Allocate device outputs.
        let mut neighbors_host = Array2::<u32>::zeros((n_q_us, k_usize));
        let neighbors_dev = ManagedTensor::from(&neighbors_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (neighbors) failed")?;

        let distances_host = Array2::<f32>::zeros((n_q_us, k_usize));
        let distances_dev = ManagedTensor::from(&distances_host)
            .to_device(&self.resources)
            .context("cuvs ManagedTensor::to_device (distances) failed")?;

        let mut search_params =
            SearchParams::new().context("cuvs SearchParams::new failed")?;
        // Ensure we always keep at least k candidates, but honor configured itopk_size.
        search_params = search_params.set_itopk_size(k_usize.max(self.itopk_size));

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

        // Copy neighbor ids back to host.
        neighbors_dev
            .to_host(&self.resources, &mut neighbors_host)
            .context("cuvs ManagedTensor::to_host (neighbors) failed")?;

        // Convert neighbor IDs to a tensor on the original query device.
        let mut flat_ids: Vec<i64> = Vec::with_capacity(n_q_us * k_usize);
        for i in 0..n_q_us {
            for j in 0..k_usize {
                flat_ids.push(neighbors_host[(i, j)] as i64);
            }
        }

        let ids_on_q = Tensor::from_slice(&flat_ids)
            .view([n_q_tokens, kd])
            .to_device(queries.device())
            .to_kind(Kind::Int64)
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
