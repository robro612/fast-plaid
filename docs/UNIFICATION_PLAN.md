# Plan: unified multi-vector retrieval architecture

Status: draft for review. No backwards-compatibility obligation.

## Goal

Build one extensible retrieval engine whose primary design target is
generality across anchoring, residual quantization, and scoring pipelines.
The goal is not a short-term merge of two codebases with compatibility shims;
the goal is a clean long-term architecture that can express:

- legacy PLAID-style IVF over k-means centroids,
- WARP-style per-cell scoring and sorted-merge aggregation,
- MaxIVF/CAGRA-style fine-grained anchors,
- future residual quantizers and scoring pipelines.

The core user-facing idea remains three major axes:

1. **Anchoring**: how anchor vectors are trained, stored, assigned, and
   selected at query time.
2. **Residual quantization**: how document-token residuals relative to anchors
   are encoded and what scoring operations the encoding supports.
3. **Scoring pipeline**: how selected anchor cells become document scores.

Internally, these axes are not perfectly independent. WARP-style scoring
requires a cell/token-oriented candidate stream and an additive residual scorer.
PLAID-style approximate MaxSim requires doc-grouped views and a query-anchor
score provider, which may be dense over all anchors or sparse over only the
anchors touched by candidate documents. CAGRA/MaxIVF avoids full query-anchor
matmul by construction. The design should make these capabilities explicit
instead of hiding them behind one over-broad trait.

## Systems being unified

We currently have two related Rust+PyO3 multi-vector retrieval engines:

- **fast-plaid**: a clean PLAID implementation using IVF over sqrt(N) k-means
  centroids, approximate centroid-code scoring, residual decompression, and
  full MaxSim reranking.
- **xtr-warp-rs**: a WARP implementation that started from fast-plaid and then
  added per-cell decompression/scoring, sorted-merge aggregation, sharding,
  mmap, tombstone delete, in-place update, compaction, and centroid expansion.

The planned architecture should also support the motivating third variant:

- **MaxIVF/CAGRA**: many fine-grained anchors trained by sampling document
  tokens, repeatedly assigning tokens to anchors with a CAGRA graph over the
  current anchors, updating anchors from their assigned tokens, and rebuilding
  the graph. The final CAGRA graph connects anchor vectors, not the full
  document-token corpus.

## High-level architecture

```text
fast_plaid::
  anchor::
    trainer.rs       # AnchorTrainer: fit initial/final anchors
    store.rs         # AnchorStore: own anchor vectors + search structures
    assigner.rs      # AnchorAssigner: document token -> anchor id(s)
    selector.rs      # AnchorSelector: query token -> candidate anchor cells
    strategy.rs      # AnchorStrategy: composed anchoring implementation
    kmeans.rs        # PLAID/WARP k-means anchoring
    maxivf_cagra.rs  # MaxIVF/CAGRA anchoring

  quantizer::
    traits.rs        # ResidualEncoder, DecodableResiduals, AdditiveResidualScorer
    scalar_per_dim.rs
    pq_normalized.rs

  scoring::
    pipeline.rs      # ScoringPipeline and pipeline capabilities
    plaid.rs         # PLAID-style dense pruning + decode + MaxSim
    warp.rs          # WARP-style cell scoring + sorted merge
    aggregation.rs   # MaxSim, per-token max, token-sum, MSE correction

  storage::
    compact.rs       # canonical anchor-grouped layout
    views.rs         # doc-grouped / anchor-unique-pids materialized views
    metadata.rs
    mmap.rs
    shard.rs

  encoding::
    sample.rs
    encode.rs

  runtime::
    device.rs
    cuda_stream.rs
    blas.rs

  bindings/lib.rs
```

## Component model

### Anchoring

Anchoring is one public axis, but internally it has four duties. These should
be independently testable and, where useful, independently swappable.

At construction time, those duties are assembled into one anchoring strategy:

```rust
struct AnchorStrategy {
    trainer: Box<dyn AnchorTrainer>,
    store: Box<dyn AnchorStore>,
    assigner: Box<dyn AnchorAssigner>,
    selector: Box<dyn AnchorSelector>,
}
```

The public API can still expose this as one `anchoring` choice. The split is
for implementation clarity: training anchors, storing/searching anchors,
assigning document tokens, and selecting query-time cells are related but not
the same responsibility.

#### `AnchorTrainer`

Duty: fit anchor vectors from document-token samples.

Outputs:

- final anchor vectors,
- trainer metadata,
- optional search/index artifacts needed by the resulting `AnchorStore`.

Implementations:

- **K-means trainer** for PLAID/WARP. Trains a relatively small centroid set,
  commonly sqrt(N)-scale.
- **MaxIVF/CAGRA trainer**. Initializes many anchors from sampled document
  tokens, builds a CAGRA graph over the current anchors, assigns training
  tokens to nearest anchors through CAGRA, updates anchors from assigned tokens,
  and repeats for a small number of iterations.

Important distinction:

- `num_training_samples` is the number of document-token vectors used to fit
  anchors.
- `num_anchors` is the number of anchor vectors/clusters being fitted.
- Initial MaxIVF anchors may be sampled real token vectors, but after updates
  they are centroid-like prototypes. The CAGRA graph is built over anchors, not
  over all training samples.

#### `AnchorStore`

Duty: own final anchor vectors and any acceleration structure needed to search
or score them.

Core operations:

```rust
trait AnchorStore: Send + Sync {
    fn num_anchors(&self) -> usize;
    fn dim(&self) -> usize;
    fn gather(&self, anchor_ids: &Tensor) -> Result<Tensor>;
    fn score_pairs(&self, query: &Tensor, token_ids: &Tensor, anchor_ids: &Tensor)
        -> Result<Tensor>;
}
```

Implementations:

- **DenseAnchorStore** for PLAID/WARP: centroid tensor resident on CPU/GPU.
  Dense query-anchor matmul is available.
- **CagraAnchorStore** for MaxIVF/CAGRA: anchor tensor plus CAGRA graph/index.
  Full dense matmul is not part of the normal query path.

#### `AnchorAssigner`

Duty: assign document-token embeddings to anchor ids during index encoding.

Core operation:

```rust
trait AnchorAssigner: Send + Sync {
    fn assign(&self, embeddings: &Tensor, store: &dyn AnchorStore) -> Result<AnchorAssignments>;
}
```

Implementations:

- **DenseNearestAssigner** for PLAID/WARP: exact nearest centroid by dense
  distance or dot-product computation.
- **CagraNearestAssigner** for MaxIVF/CAGRA: approximate nearest-anchor search
  through CAGRA. This is what makes very large anchor sets practical.

Version-1 policy: single-anchor assignment per document token. The types should
leave room for multi-anchor assignment later, but v1 should not pay that
complexity tax unless benchmarks require it.

#### `AnchorSelector`

Duty: select candidate anchor cells for query tokens during search.

Core operation:

```rust
trait AnchorSelector: Send + Sync {
    fn select(
        &self,
        query: &Tensor,
        query_mask: Option<&Tensor>,
        store: &dyn AnchorStore,
        ctx: &SelectionContext,
    ) -> Result<CellSelection>;
}
```

`CellSelection` should be explicitly cell/token-oriented, because WARP needs
that structure:

```rust
struct CellSelection {
    anchor_ids: Tensor,        // [num_selected_cells]
    token_ids: Tensor,         // [num_selected_cells]
    anchor_scores: Tensor,     // q_token dot anchor
    mse_estimate: Option<Tensor>,
}
```

Implementations:

- **PlaidDenseSelector**: dense `Q @ centroids.T`, top `nprobe`, usually
  followed by deduping anchors and aggregating candidate docs.
- **WarpDenseSelector**: dense per-token topk with WARP knobs: threshold,
  over-fetch bound, adaptive `t_prime`, and dummy centroid handling.
- **CagraSelector**: CAGRA top-k graph traversal per query token. It avoids
  dense `Q @ anchors.T`, which is the whole point of large MaxIVF anchor sets.

### Residual quantization

Residual quantizers should not be forced into one trait that every
implementation must fully support. Instead, use capability traits.

#### `ResidualEncoder`

Duty: train and encode residuals relative to assigned anchors.

```rust
trait ResidualEncoder: Send + Sync {
    fn train(
        &mut self,
        embeddings: &Tensor,
        anchors: &Tensor,
        assignments: &AnchorAssignments,
    ) -> Result<()>;

    fn encode(
        &self,
        embeddings: &Tensor,
        anchors: &Tensor,
        assignments: &AnchorAssignments,
    ) -> Result<EncodedResiduals>;

    fn storage_layout(&self) -> &[ResidualArraySpec];
}
```

#### `DecodableResiduals`

Duty: reconstruct approximate document-token embeddings for full MaxSim.

```rust
trait DecodableResiduals: ResidualEncoder {
    fn decode(&self, encoded: &EncodedResidualsView, anchors: &Tensor) -> Result<Tensor>;
}
```

PLAID-style reranking requires this capability.

#### `AdditiveResidualScorer`

Duty: score encoded residuals without full reconstruction, assuming the
residual contribution decomposes into independent groups.

```rust
trait AdditiveResidualScorer: ResidualEncoder {
    fn precompute_group_scores(&self, query_token: &Tensor) -> Result<GroupScoreTable>;

    fn score_codes(
        &self,
        encoded: &EncodedResidualsView,
        group_scores: &GroupScoreTable,
        aux: &ResidualAuxView,
    ) -> Result<Tensor>;
}
```

WARP-style per-cell scoring requires this capability.

The "groups" are not required to be individual dimensions. The requirement is
that the residual score can be written as a sum over independently encoded
groups:

```text
q · residual ~= sum_g group_score[g, code_g]
```

For scalar-per-dim quantization, `g` is one embedding dimension. For PQ,
`g` is one subspace. PQ therefore can still support WARP-style scoring, but the
score table is indexed by subspace code rather than by dimension bucket. For
PQ-normalized residuals, the per-row residual magnitude is an additional
payload column that must be multiplied into the residual term.

Implementations:

- **ScalarPerDimQuantizer**: current residual codec. Groups are dimensions,
  codes are 2-bit or 4-bit bucket ids, and weights are learned quantile bucket
  representatives. Supports both `DecodableResiduals` and
  `AdditiveResidualScorer`.
- **PQNormalizedQuantizer**: residual direction is normalized, encoded by PQ
  subspace codebooks, and paired with a stored magnitude. Supports decode.
  It can support additive scoring if the scorer receives the per-row magnitude
  auxiliary array and applies it to the residual term.

### Scoring pipelines

Scoring should be modeled as a pipeline rather than one universal `Refiner`.
The pipeline declares which selection shape, storage views, and quantizer
capabilities it requires.

```rust
trait ScoringPipeline: Send + Sync {
    fn required_views(&self) -> &[StorageViewKind];
    fn required_quantizer_caps(&self) -> QuantizerCaps;
    fn required_selection_caps(&self) -> SelectionCaps;

    fn search(
        &self,
        query: &Tensor,
        selection: &CellSelection,
        anchors: &dyn AnchorStore,
        quantizer: &dyn ResidualEncoder,
        storage: &IndexStorage,
        k: usize,
    ) -> Result<SearchResult>;
}
```

Pipelines should be composable around candidate-set boundaries:

```text
CellSelection
  -> CandidateSet              # pids, optional sparse evidence
  -> ScoredCandidateSet        # pids + approximate scores, optional
  -> final scorer
```

This allows short-circuit variants. A pipeline may return candidates directly
after anchor-to-posting expansion, after approximate MaxSim, or after a final
scorer. A no-quantization index can attach a full-embedding MaxSim scorer after
either candidate boundary, using a `DocEmbeddingView` instead of residual
decode.

#### Candidate generation to approximate MaxSim

The initial candidate-generation path should be reusable across dense IVF,
WARP-style selection, and CAGRA selection. The selector chooses cells; storage
views and scoring stages decide how those cells become document scores.

```text
AnchorSelector
  -> CellSelection

AnchorPidView
  -> candidate pids

DocPostingView
  -> candidate docs' ordered anchor ids

AnchorScoreProvider
  -> Q @ union(candidate doc anchors).T

ApproxMaxSimScorer
  -> approximate document scores
```

The responsibilities are:

1. **`AnchorSelector`** emits selected `(query_token_id, anchor_id)` cells. The
   implementation can be dense top-k over `Q @ anchors.T` or CAGRA graph
   lookup.
2. **`AnchorPidView`** expands selected anchor ids to candidate passage ids.
   This view only needs unique pids per anchor; it does not need residuals.
3. **`DocPostingView`** loads the candidate documents' ordered token metadata:
   `doc_lengths` and flattened `doc_anchor_ids` ordered by pid and original
   document-token order.
4. **`AnchorScoreProvider`** computes or retrieves query-anchor scores for the
   union of anchors touched by those candidate documents. Legacy PLAID can
   implement this as `Q @ all_anchors.T`; sparse/CAGRA-compatible PLAID can
   compute only `Q @ union_doc_anchors.T`.
5. **`ApproxMaxSimScorer`** remaps each doc token's anchor id into the local
   score table, pads by `doc_lengths`, and runs centroid-only MaxSim.

This generalizes PLAID's current approximate scoring stage. The current
implementation computes the maximal dense table, then gathers scores by
doc-token anchor id. The generalized version allows a sparse score table over
only the anchors touched by candidate docs.

#### PLAID pipeline

Duty: approximate document pruning followed by full residual decompression and
MaxSim.

Stages:

1. Select candidate cells via an `AnchorSelector`.
2. Gather candidate passage ids through `AnchorPidView`.
3. Approximate-score documents with `DocPostingView`, `AnchorScoreProvider`,
   and `ApproxMaxSimScorer`.
4. Keep top candidates.
5. Decode residuals for candidates.
6. Compute full token-token scores and MaxSim.

Requirements:

- candidate discovery through `AnchorPidView`,
- doc-grouped `DocPostingView`,
- query-anchor scores from `AnchorScoreProvider`,
- `DecodableResiduals`,
- residual payloads for final rerank.

This pipeline is natural for k-means IVF. With CAGRA/MaxIVF it should use the
sparse `AnchorScoreProvider` form, because a dense all-anchor matmul conflicts
with the reason for using graph search.

#### WARP pipeline

Duty: score selected cells directly and aggregate by sorted merge.

Stages:

1. Select per-token cells.
2. For each selected cell, compute per-candidate score:
   `q_token dot anchor + residual_score`.
3. For each token, merge that token's `nprobe` cell lists by max over duplicate
   passage ids.
4. Merge token-level lists by sum across tokens with MSE correction.
5. Return top-k.

Requirements:

- cell/token-oriented `CellSelection`,
- anchor-grouped compact storage,
- `AdditiveResidualScorer`,
- per-cell candidate lists sorted by passage id.

The current WARP implementation specializes this residual scoring loop to
scalar-per-dimension buckets. In the unified design, the loop should be over
quantizer-defined `(group, code)` pairs instead:

```text
residual_score = sum_g group_score[g, code_g]
```

For scalar quantization, `g` is a dimension. For PQ, `g` is a subspace and
`code_g` selects a subspace codeword. The WARP pipeline should not know which
case it is handling; it should call `AdditiveResidualScorer`.

This pipeline is the best fit for both current WARP and MaxIVF/CAGRA, because
it stays cell-local after selection and works naturally over anchor-grouped
storage.

#### CAGRA + WARP pipeline

This is not a separate scoring algorithm; it is the WARP pipeline paired with:

- `MaxIvfCagraTrainer`,
- `CagraAnchorStore`,
- `CagraNearestAssigner`,
- `CagraSelector`,
- a quantizer that supports additive residual scoring.

The important constraint is that CAGRA changes anchoring and selection, while
WARP supplies the cell-oriented scoring and aggregation.

## Component instantiations

| Component | PLAID | WARP | MaxIVF/CAGRA + WARP |
|---|---|---|---|
| AnchorTrainer | K-means over training samples, sqrt(N)-scale centroids | K-means over training samples, same basic centroid training | Sample many token anchors, CAGRA assignment/update loop, rebuild graph over anchors |
| AnchorStore | Dense centroid tensor | Dense centroid tensor plus per-centroid sizes used by selector/runtime | Final anchor tensor plus CAGRA graph/index over anchors |
| AnchorAssigner | Exact dense nearest-centroid assignment | Exact dense nearest-centroid assignment | CAGRA nearest-anchor assignment |
| AnchorSelector | Dense top-`nprobe` over `Q @ centroids.T` | Dense per-token topk with threshold, bound, `t_prime`, dummy centroid | CAGRA top-k per query token |
| Canonical storage | Historically doc-grouped + IVF; unified plan derives this from anchor-grouped storage | Anchor-grouped compacted arrays | Anchor-grouped compacted arrays |
| Quantizer | Scalar-per-dim today | Scalar-per-dim today | PQ-normalized target, scalar-per-dim possible baseline |
| Required quantizer capability | Decode | Additive residual scoring | Additive residual scoring |
| ScoringPipeline | PLAID dense pruning + decode + MaxSim | WARP per-cell scoring + sorted merge | WARP per-cell scoring + sorted merge |

## Storage model

The logical source of truth is a stream of postings:

```text
(pid, anchor_id, residual_code, quantizer_aux...)
```

Physical views should separate indexing from payload columns. This lets a
deployment store only the views needed by its configured scoring pipelines.

Canonical storage should be anchor-grouped for WARP and MaxIVF/CAGRA:

- `pids.compacted.npy`: passage id per encoded embedding, sorted by anchor id.
- `sizes.compacted.npy`: number of embeddings per anchor.
- `offsets.compacted.npy`: anchor offsets into compacted arrays.
- quantizer arrays declared by `storage_layout()`, such as
  `residuals.compacted.npy`, `pq_codes.compacted.npy`, or
  `magnitudes.compacted.npy`.

Views are materialized only when a pipeline requests them:

| View | Needed by | Notes |
|---|---|---|
| `AnchorPidView` | candidate discovery | `anchor_id -> unique pids`; no residual payload required. |
| `AnchorPostingView` | WARP cell scoring | `anchor_id -> posting rows`, with pids and residual/aux payloads aligned in anchor order. |
| `DocPostingView` | PLAID approximate scoring + rerank | `pid -> ordered token rows`; token position is implicit in row order. |
| `DocEmbeddingView` | no-quantization full MaxSim | `pid -> ordered full token embeddings`; bypasses residual decode. |

`DocPostingView` should store rows in original document-token order, so it does
not need an explicit `token_pos` column. `AnchorPostingView` only needs token
positions if we want to derive doc-ordered outputs from anchor-grouped rows
after the fact. If a doc view is materialized during build, token order can be
preserved there directly.

## Configuration sketch

```python
fp = FastPlaid(
    index_path="...",
    anchoring={
        "trainer": {
            "kind": "maxivf_cagra",
            "num_training_samples": 50_000_000,
            "num_anchors": 25_000_000,
            "iterations": 5,
            "graph_degree": 64,
        },
        "assigner": {"kind": "cagra_nearest"},
        "selector": {"kind": "cagra", "nprobe": 4},
    },
    quantizer={
        "kind": "pq_normalized",
        "subspaces": 32,
        "bits_per_subspace": 8,
        "store_magnitude": "fp16",
    },
    scoring={"kind": "warp"},
)
```

The public API can expose this as one `anchoring` block, while the Rust side
constructs the internal trainer/store/assigner/selector components.

## Rust design notes

Avoid generic methods on traits intended for trait objects. For example, this
is not object-safe:

```rust
trait Refiner {
    fn refine<A: Anchor, Q: Quantizer>(&self, anchor: &A, quantizer: &Q);
}
```

Prefer either sealed enums for known implementations or object-safe traits with
capability checks:

```rust
fn search(
    &self,
    anchors: &dyn AnchorStore,
    quantizer: &dyn ResidualEncoder,
    storage: &IndexStorage,
) -> Result<SearchResult>;
```

When a pipeline needs `DecodableResiduals` or `AdditiveResidualScorer`, perform
that dispatch explicitly through enums or object-safe downcasting/capability
wrappers. The key is to make unsupported combinations fail at construction time,
not halfway through a search.

## Supported combinations

First-class v1 combinations:

1. **PLAID legacy**
   - K-means anchoring.
   - Scalar-per-dim quantizer.
   - PLAID scoring pipeline.

2. **WARP legacy**
   - K-means anchoring.
   - Scalar-per-dim quantizer.
   - WARP scoring pipeline.

3. **MaxIVF/CAGRA + WARP target**
   - MaxIVF/CAGRA anchoring.
   - PQ-normalized quantizer if additive scoring validates well; otherwise
     scalar-per-dim as a baseline.
   - WARP scoring pipeline.

Deferred or experimental combinations:

- **CAGRA + PLAID dense pruning**: likely not a good fit because PLAID's dense
  approximate stage requires the full query-anchor score matrix.
- **Multi-anchor document-token assignment**: useful possible recall win, but
  changes storage, scoring, and duplicate handling.
- **Non-additive quantizers with WARP**: unsupported unless they provide a
  different cell scorer.

## Open design questions

1. Should `AnchorSelector` always emit token ids, even for PLAID? Doing so
   simplifies the common `CellSelection` type and keeps WARP's needs explicit.
2. Should token positions be stored in canonical storage, or only in a derived
   view for APIs that return token-score matrices?
3. Should quantizer dispatch use sealed enums rather than trait objects for
   better Rust ergonomics and fewer runtime capability checks?
4. For MaxIVF/CAGRA, do we rebuild the CAGRA graph every refinement iteration,
   or update it in-place if cuVS exposes an efficient path?
5. What is the first validation target for PQ-normalized additive scoring:
   k-means + WARP, or MaxIVF/CAGRA + WARP directly?

