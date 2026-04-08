# Coding Task: Add a Novel, Rate-Aware DT-CWT Control Path and Clean Up Research Reliability

## Goal

This repo already has a working 3D DT-CWT video preprocessing pipeline, but its core decision rule is still close to BayesShrink-style adaptive thresholding.  
Your task is to introduce a **more novel control principle** while keeping the current pipeline usable.

The first target is:

> **Rate-aware, content-aware subband threshold control**  
> instead of using only noise/subband variance.

Do **not** rewrite the whole project.  
Build on the current structure and keep CPU/GPU parity as much as possible.

---

## Primary Objective

Implement a new preprocessing mode:

- `rate_aware`
- works for both CPU and CUDA paths
- adjusts threshold strength using:
  - estimated noise level
  - motion strength
  - edge/texture density
  - target bitrate
- can be turned on/off from CLI
- preserves the current baseline behavior when disabled

This should be implemented as a **new controller layer on top of the existing DT-CWT shrinkage**, not as a separate unrelated pipeline.

---

## Why this change

The current implementation appears to rely on adaptive thresholding based mainly on estimated noise and subband variance.  
That is useful, but still conventional.

The new contribution should shift the method from:

- “DT-CWT + standard adaptive shrinkage”

to:

- “**rate-conditioned spatio-temporal wavelet control for pre-encoding**”

That is the research direction we want.

---

## Scope

### In scope

1. Add a context-aware threshold controller
2. Wire pipeline context into the processor
3. Preserve old behavior as default
4. Add minimal tests
5. Add logging/reporting hooks so experiments can compare modes
6. Clean up obvious reliability issues that block valid experiments

### Out of scope

1. Full learned model training
2. Major repo restructuring
3. New dataset download automation
4. Full packaging rewrite
5. Large-scale benchmark execution

---

## Files you are expected to modify

Prioritize these files:

- `dtcwt_processor.py`
- `dtcwt_cuda.py`
- `main_pipeline.py`
- `run_noise_experiment.py`
- `run_rd_curve.py`

You may also touch these if needed:

- `evaluate_metrics.py`
- `advanced_evaluation.py`
- `test.py`
- `test_caching_math.py`
- `pyproject.toml`
- `README.md` or `read.md`

Do not create unnecessary new modules unless they make the control logic much cleaner.

---

## Required implementation details

## 1. Add a processing context object

Add a lightweight processing context that can be passed into the DT-CWT processor for each chunk.

Suggested fields:

- `target_bitrate_kbps: float`
- `noise_level: float`
- `motion_strength: float`
- `edge_density: float`
- `scene_cut: bool`
- `chunk_index: int`
- `fps: float | None`
- `mode: str`  (`baseline`, `adaptive`, `rate_aware`)

You may implement this as either:

- a small dataclass, or
- a plain dict with validation

Preferred: dataclass.

Name suggestion:

- `ProcessingContext`

---

## 2. Add processor-side context setter

In `DTCWT3DProcessor`, add something like:

- `set_context(context: ProcessingContext)`

Store the latest context on the processor instance.

The processor must still work if no context is provided.

In that case, behavior must fall back to the current logic.

---

## 3. Add new threshold mode: `rate_aware`

### Existing behavior to preserve

Keep current behavior available as-is:

- fixed threshold path
- current adaptive threshold path

### New behavior to add

Add a third threshold mode:

- `threshold_mode="rate_aware"`

Suggested API:

- `threshold_mode: str = "adaptive"`

Accepted values:

- `fixed`
- `adaptive`
- `rate_aware`

Do not break old code that only uses `adaptive_threshold=True/False`.  
You may internally translate old flags into the new mode, but maintain backward compatibility.

---

## 4. Implement the controller formula

Implement a simple but explicit controller multiplier on top of the current adaptive threshold.

Current adaptive threshold already looks roughly like:

- estimate noise
- estimate subband signal variance
- compute adaptive threshold
- shrink magnitude

Keep that structure.

Then add a multiplier:

`T_final = T_adapt * controller_multiplier`

Where `controller_multiplier` is based on chunk context.

### Required design intent

The multiplier should increase threshold strength when:

- target bitrate is lower
- estimated noise is higher
- content is relatively smooth / low-detail

The multiplier should reduce threshold strength when:

- motion is stronger
- edge density / texture is higher
- preserving detail is more important

### Suggested initial form

Use a bounded multiplicative rule such as:

`mult = exp(a * noise_term + b * bitrate_term - c * motion_term - d * edge_term)`

Then clamp to a safe range, for example:

- min: `0.5`
- max: `2.5`

You may choose another smooth bounded function if well justified.

### Normalization requirement

All context features must be normalized into stable ranges.

For example:

- bitrate term should not depend on string parsing every time
- motion and edge terms should be scaled so they do not explode
- noise estimate should be numerically stable across clips

Document the normalization in comments.

---

## 5. Compute chunk-level features in `main_pipeline.py`

Before calling `processor.process_chunk(...)`, compute chunk-level features from the current Y chunk.

Implement at least:

### A. Noise estimate
A robust scalar estimate for current chunk noise.

Acceptable options:

- MAD-based estimate on high-frequency residual proxy
- frame-difference + local texture suppression
- simple Laplacian-based estimator

Keep it lightweight.

### B. Motion strength
Estimate temporal activity across frames.

Acceptable options:

- mean absolute frame difference
- normalized temporal gradient magnitude

Keep it cheap. No optical flow.

### C. Edge density
Estimate spatial detail level.

Acceptable options:

- Sobel magnitude percentile mask ratio
- Laplacian energy density

Again, keep it lightweight.

### D. Bitrate context
Parse bitrate input once and expose it numerically in kbps.

---

## 6. Pass context into processor

For each chunk in `main_pipeline.py`:

1. compute features
2. create `ProcessingContext`
3. pass it to processor
4. run preprocessing

Suggested flow:

- `context = build_processing_context(...)`
- `processor.set_context(context)`
- `processed_y = processor.process_chunk(...)`

---

## 7. Maintain CPU/GPU parity

The new controller must affect both:

- CPU shrinkage path in `dtcwt_processor.py`
- CUDA shrinkage path in `dtcwt_cuda.py`

The formulas do not need to be bit-exact, but they should be **functionally consistent**.

Do not implement the new logic only on CPU.

---

## 8. Add CLI controls

Add CLI arguments in `main_pipeline.py`:

- `--threshold-mode {fixed,adaptive,rate_aware}`
- `--controller-a`
- `--controller-b`
- `--controller-c`
- `--controller-d`
- `--min-multiplier`
- `--max-multiplier`

Optional but recommended:

- `--log-context`
- `--disable-rate-aware-scene-reset`

Keep defaults reasonable so the pipeline still works out of the box.

---

## 9. Add experiment logging

When enabled, log one record per chunk with:

- chunk index
- overlap length
- bitrate
- noise estimate
- motion strength
- edge density
- controller multiplier
- threshold mode

CSV is preferred.

Suggested output path:

- `logs/context_log_<video>_<bitrate>.csv`

This is important for later analysis and ablation.

---

## 10. Add minimal ablation support

Update experiment scripts so they can compare at least:

- baseline x264
- current adaptive DT-CWT
- new rate-aware DT-CWT

Do not redesign all experiment scripts.  
Just add enough hooks so the new mode can be invoked cleanly.

If you need a single canonical flag, use:

- `--threshold-mode rate_aware`

---

## Reliability fixes you should also do

These are not the main novelty, but they matter because they affect experimental validity.

### A. Fix overlap state initialization robustly

Make sure `prev_y_overlap` is always initialized safely in `read_y4m_and_split(...)`.

Even if this was partially fixed already, make the function robust and explicit.

### B. Remove duplicated custom metric logic if practical

`compute_custom_metrics(...)` appears duplicated across evaluation files.

If possible without causing breakage:

- consolidate shared logic into one implementation
- keep public interfaces stable

If consolidation would risk breaking current scripts, at minimum add a TODO and align behavior/comments.

### C. Keep current defaults backward compatible

Current scripts and one-off experiments should still run without needing all the new flags.

---

## Optional stretch goal

If time permits, add a very small scene-aware guard:

- if `scene_cut == True`, either:
  - clamp the multiplier toward neutral, or
  - skip aggressive threshold inflation for that chunk

This is optional, but useful.

---

## Testing requirements

Add lightweight tests or validation scripts for the following:

### 1. Backward compatibility
- old invocation still runs
- `adaptive` mode still behaves close to the previous implementation

### 2. New mode activation
- `rate_aware` mode changes the effective threshold when context changes

### 3. CPU/GPU parity sanity
- no crash on either path
- same context can be passed into both

### 4. Robust overlap behavior
- no uninitialized overlap state error
- overlap length 0 still works

### 5. Logging
- CSV log is written when enabled
- required columns exist

Do not spend time building a huge test framework.  
Small, focused tests are enough.

---

## Acceptance criteria

Your work is complete only if all of the following are true:

1. A new `rate_aware` mode exists
2. It uses bitrate + motion + edge + noise context
3. CPU and CUDA paths both support it
4. `main_pipeline.py` can invoke it from CLI
5. The old adaptive mode still works
6. Chunk-level context can be logged
7. The overlap-state bug is robustly handled
8. The diff is readable and documented with comments where the new logic is non-obvious

---

## Implementation notes

### Important constraints

- Keep the code readable
- Prefer explicit math over clever abstractions
- Add comments for normalization choices
- Avoid expensive feature extraction
- Do not introduce heavy new dependencies
- Do not silently change experimental semantics without documenting it

### Performance guidance

The controller feature extraction should be cheap relative to DT-CWT itself.

Use lightweight per-chunk statistics only.

No optical flow.  
No deep model.  
No expensive per-frame perceptual pre-analysis.

---

## Suggested development order

1. Add `ProcessingContext`
2. Add processor `threshold_mode`
3. Implement CPU controller
4. Implement CUDA controller
5. Compute features in pipeline
6. Add CLI flags
7. Add CSV logging
8. Fix overlap robustness
9. Add minimal tests
10. Update README / comments

---

## Deliverables

At the end, provide:

### 1. Code changes
Normal git diff / modified files.

### 2. Short implementation summary
Explain:
- what was changed
- where the controller is applied
- how features are computed
- how backward compatibility was preserved

### 3. Validation summary
State:
- what commands were run
- what passed
- what was not fully validated

### 4. Known limitations
Be explicit about:
- controller heuristic nature
- lack of learned tuning
- any CPU/GPU numeric differences
- any evaluation script limitations still left

---

## Nice-to-have naming

If you need a concise research-facing name, use one of these consistently:

- `RateAwareDT-CWT`
- `Rate-Conditioned DT-CWT`
- `Context-Aware DT-CWT Prefilter`

Preferred internal mode name:

- `rate_aware`

---

## Do not do these

- Do not replace DT-CWT with a different method
- Do not hardcode video-specific constants without comments
- Do not remove current experiment paths
- Do not add a heavyweight ML training pipeline
- Do not “clean up” the entire repo as a side quest
- Do not make packaging changes unless necessary for the task

---

## Final success condition

After your changes, I should be able to say:

> “This repo no longer just applies standard adaptive wavelet shrinkage.  
> It now performs **rate-aware, content-aware spatio-temporal control** before encoding, and the implementation is usable in the existing research pipeline.”
