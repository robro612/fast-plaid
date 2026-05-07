//! cuVS CAGRA `IndexParams`: the public Rust wrapper does not expose `set_metric`, but the
//! underlying C struct has `metric`. ColBERT-style usage is cosine on L2-normalized vectors;
//! set **`CosineExpanded`** so graph build and search use the matching distance family (default
//! is L2-expanded).

#![cfg(feature = "cagra")]

use cuvs::cagra::IndexParams;
use cuvs::distance_type::DistanceType;

pub fn cagra_index_params_set_cosine_expanded(params: &mut IndexParams) {
    unsafe {
        (*params.0).metric = DistanceType::CosineExpanded;
    }
}
