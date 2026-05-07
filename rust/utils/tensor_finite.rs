//! Lightweight checks that tensors are finite before calling foreign code (e.g. cuVS).

use anyhow::{anyhow, Result};
use tch::{Kind, Tensor};

/// Fails if `t` contains NaN or infinity when viewed as float32.
///
/// Graph construction usually ingests finite training data, but **queries** are a separate
/// path and bad values can still reach CAGRA search kernels and provoke undefined behavior.
pub fn ensure_tensor_all_finite(t: &Tensor, context: &str) -> Result<()> {
    let ok = t
        .shallow_clone()
        .detach()
        .to_kind(Kind::Float)
        .isfinite()
        .all();
    if ok.numel() != 1 {
        return Err(anyhow!("{context}: internal error, expected scalar from isfinite().all()"));
    }
    if ok.double_value(&[]) == 0.0 {
        return Err(anyhow!(
            "{context}: tensor contains nan/inf — refusing to pass to cuVS CAGRA"
        ));
    }
    Ok(())
}
