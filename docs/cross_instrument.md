<!--
cross_instrument.md
Anna Preto
AstroDeep Project
Runbook of the Rubin x Euclid cross-instrument work on Jean Zay: datasets D1 and D2, Stage 1, Stage 2.
-->

# Rubin × Euclid cross-instrument learning — runbook (Jean Zay)

Everything here runs on Jean Zay. Nothing is meant to run on a laptop: the Euclid images alone do not fit.
The DebVAder reference run is **given on the command line** (`DEBVADER_RUN`), never written in a file.

**Reference VAE (2026-11-17).** `paper_baseline` was trained on noiseless BTK simulations (`generate_dataset.py`,
regime `clean`). Phase 0 on 3000 DP2 stamps: units right (flux within 16 % at S/N 100–1000), but the sky noise is
read as diffuse light — a reconstructed plateau of ~4σ over the whole stamp (i band: 2540 e⁻ per pixel at the edge,
DP2 sky σ 670), reconstructed flux +1560 % at S/N 5–20, median χ² per pixel 13.5 at every S/N. Its latent space
cannot be the reference for real stamps. The reference is now a DebVAder stage 1 **trained from scratch on the
Rubin stamps of D1** (step 3b below), same architecture, debvader-keras' own trainer; `paper_baseline` only serves
as the architecture template of that run.

## What is built

| step | command | needs | output under `DP2_ML_ROOT` |
|---|---|---|---|
| MMU mirror | `dp2-ml <cfg> mmu-mirror [--dry-run]` | internet (prepost), ~30 GB per partition | `euclid.mmu_cache` (`$SCRATCH/mmu_euclid_q1`) |
| probe | `dp2-ml <cfg> probe-euclid --patch T_P` | internet (prepost) | `evaluations/probe_euclid/` |
| phase 0 | `dp2-ml <cfg> phase0` | final stamp dataset, VAE | `evaluations/phase0/` |
| D1 | `dp2-ml <cfg> d1 --index N`, then `d1-merge` | final stamp dataset, internet | `datasets/d1/` |
| D2 | `dp2-ml <cfg> d2 --index N`, then `d2-merge` | per-patch copaings + images, internet, D1 merged | `datasets/d2/` |
| quality | `dp2-ml <cfg> quality d1` / `quality d2` | merged dataset | `datasets/d*/quality_report.json` |
| blends | `dp2-ml <cfg> blends` | D1 | `datasets/blends/` |
| Rubin VAE set | `dp2-ml <cfg> export-rubin-vae` (`--debvader-run` = template) | D1 | `datasets/rubin_vae/` |
| Rubin VAE | `scripts/jeanzay/ml/rubin_vae.slurm` (debvader-keras `train.py --stage 1`) | the set, GPU | `$EXPERIMENTS_DIR/<rubin_vae.run_name>/` |
| Stage 1 | `train-stage1 --ablation A`, `eval-stage1 --ablation A` | D1, VAE, GPU | `runs/A/`, `evaluations/A/` |
| Stage 2 | `train-stage2 --ablation B --stage1-run A`, `eval-stage2 ...` | D2, blends, Stage 1 run, GPU | `runs/B/`, `evaluations/B/` |
| notebooks | `scripts/jeanzay/ml/notebooks.slurm` | the above | `notebooks/` (executed copies with figures) |

- **D1**: one pair per galaxy of the final stamp dataset, with a Euclid cutout of the same galaxy around its Euclid
  position (119 × 119 px at 0.1″). No re-selection: unusable pairs go to `rejected.parquet` with their reason;
  every other check is a column.
- **D2**: Euclid targets of copaings groups with 2 Euclid / ≥ 1 Rubin objects, Rubin side blended (`sb_rubin > 0`),
  alone at Euclid resolution, and whose shape, centroid, size and flux barely change when the other Euclid
  sources are masked. The Rubin stamp is the unmasked blend, centred on the Euclid target (`rubin_*`) and on the Rubin
  object (`rubin_*_rc`).
- **Splits**: by Rubin `(tract, patch)` through a hash, the same for every dataset: no patch feeds two splits, and
  no Stage 2 test patch was seen by the Stage 1 teacher. `splits.fixed` overrides the draw for named patches: the 9
  patches of the first scale run drew no test patch, so `5063_45` is set to test (val: `5063_33`, by the draw).
- **Rubin VAE set**: the Rubin stamps of D1 in debvader-keras' real-build format (`images.npy` in nJy, `meta.npz`
  with the D1 split, `stats.json`, `debvader_config.json`). Masked pixels get Gaussian noise at the sky level of
  their stamp and band, so the VAE never sees holes; stamps below `rubin_vae.min_valid_fraction` are left out. The
  VAE is trained on the D1 train split only, early-stopped on val: test patches stay unseen by every network.
- **Stage 1**: Euclid encoder trained to land where the frozen DebVAder encoder puts the Rubin view
  (latent loss), and so that the frozen decoder gives back the Rubin stamp (cross loss). Ablations in
  `stage1.ablations`.
- **Stage 2**: Rubin blend encoder (DebVAder encoder architecture, warm-started from the reference encoder,
  `stage2.init`) trained to return the teacher latent of the target. Baselines in the evaluation: the
  isolated-galaxy encoder fed the blend, and the untouched DebVAder deblender when the reference run has a stage 2
  (the D1-trained run has none).

## Installation on Jean Zay (once)

One environment holds both packages (the ML jobs import both):

```bash
module purge; module load arch/a100   # on an A100 node; skip on the login node if building CPU-only first
python -m venv $WORK/venvs/dp2-ml && source $WORK/venvs/dp2-ml/bin/activate
pip install -e "<debvader-keras clone>[jax-cuda]"
pip install -e "<rubin-euclid-stamps clone>[ml,test]"
pip install --no-deps "friendly @ git+https://github.com/LSSTDESC/friendly"
pytest <rubin-euclid-stamps clone>/tests          # on a node with the environment; test_ml.py needs keras
```

## Environment of every submission

```bash
cd <rubin-euclid-stamps clone>
export DP2_ML_VENV=$WORK/venvs/dp2-ml
export DP2_CONFIG=config/<patch-by-patch pipeline config>.yaml   # the one that assembled the final dataset
export DP2_ML_CONFIG=config/ml_jeanzay.yaml
export DP2_ML_ROOT=$WORK/dp2-ml
export DP2_CPU_ACCOUNT=<projet>@cpu DP2_GPU_ACCOUNT=<projet>@a100
export DEBVADER_RUN=<folder of the DebVAder run>         # contains config.json and checkpoints/
# export DEBVADER_STATS=<stats.json of its training set>   # only if neither the run nor its dataset has one
export DEBVADER_REPO=$WORK/debvader-keras                 # checkout with scripts/train.py (step 3b)
export EXPERIMENTS_DIR=$WORK/debvader-results             # where debvader-keras writes its runs (step 3b)
```

`DEBVADER_RUN` is `paper_baseline` up to step 3b (template of the architecture), then the D1-trained run
`$EXPERIMENTS_DIR/<rubin_vae.run_name>` for every later step. `rubin.vae_units: auto` takes the units from the run
(nJy for the D1-trained run, BTK electrons for paper_baseline); an explicit value that disagrees stops the job.

To find the run folder: it holds `config.json` and `checkpoints/stage1_best.weights.h5`, e.g.
`find $WORK $SCRATCH -path '*paper_baseline*' -name stage1_best.weights.h5 2>/dev/null`.

## Order, and what to look at before going on

0. **MMU mirror** — each partition of the MMU Euclid catalogue is a single parquet row group of ~30 GB of
   compressed images (probe of 2026-11-18): no smaller piece can be fetched, and lsdb loads whole partitions in
   memory. `dp2-ml $DP2_ML_CONFIG mmu-mirror --dry-run` (login node) lists the partitions overlapping the patches
   and their size; `scripts/jeanzay/ml/submit_ml.sh mmu-mirror` downloads them once (resumable, files already
   complete are skipped) into `euclid.mmu_cache`. The probe and D1/D2 then stream the local files batch by batch
   (memory: `euclid.mmu_batch_rows` rows). `$SCRATCH` purges files untouched for 30 days: the readers check sizes
   and ask for a rerun of the mirror rather than reporting missing images.
1. **Probe** — `scripts/jeanzay/ml/submit_ml.sh probe <tract_patch>` (after the mirror).
   `evaluations/probe_euclid/probe_<patch>.json` says whether MMU (Hugging Face) or IRSA answers from prepost,
   and shows the raw structure of the MMU image column (band names, sizes, scale). If the band names differ from
   `MMU_BANDS` in `sources/euclid_images.py`, extend that table. If neither works, the Euclid cutouts must be
   fetched elsewhere and copied: say so before going further.
   Look also at `source_px` of each MMU cutout: if it is smaller than `euclid.stamp_px` (119), the missing border is
   masked and every pair falls below `dataset1.min_valid_fraction_euclid`. Then run with
   `--set euclid.stamp_px=<odd value <= source_px>`, or set it in the configuration file.
2. **Phase 0** — `submit_ml.sh phase0`. In `evaluations/phase0/phase0_report.json`:
   `decoder_shared` must be true, `deterministic` true; the DP2 normalised peaks should sit near the training value
   (0.92) and the fraction of pixels above saturation near 0. `domain_gap_by_snr` measures how well a VAE trained
   on BTK simulations reconstructs DP2 galaxies: it is the floor every later number is read against.
   If `reference` reproduction fails, stop.
3. **Datasets** — `submit_ml.sh datasets`. Read `datasets/d1/quality_report.json` and notebook
   `notebooks/ml/01`: astrometry (systematic shift ≪ 1 Rubin pixel), e1/e2 correlation between VIS and Rubin
   (positive), position-angle table (set `euclid.pa_offset_deg` to the convention with the smallest median
   difference, then rebuild D2 with `--force`), split leakage.
3b. **Rubin VAE** — with `DEBVADER_RUN` = paper_baseline: `submit_ml.sh rubin-vae` (export on CPU, then training
   and phase 0 of the new run on one A100; resubmit `rubin_vae.slurm` as is if it hits 20 h, `--resume` continues).
   Read `datasets/rubin_vae/info.json` (counts per split, filled fraction) and
   `evaluations/phase0_<run>/phase0_report.json`: χ² per pixel near 1, no reconstructed plateau, flux and
   ellipticity residuals centred at every S/N. Then `export DEBVADER_RUN=$EXPERIMENTS_DIR/<run>`.
4. **Stage 1** — `submit_ml.sh stage1` (every ablation) or `submit_ml.sh stage1 s1_latent_cross`.
   `evaluations/<run>/stage1_test_summary.json` holds the gate. **Go to Stage 2 only if it passes**, or after
   deciding explicitly which criterion is relaxed and why.
5. **Stage 2** — `STAGE1_RUN=<chosen Stage 1 run> submit_ml.sh stage2`.
6. **Notebooks** — `STAGE1_RUN=... STAGE2_RUN=... submit_ml.sh notebooks`; executed copies in
   `$DP2_ML_ROOT/notebooks/`, to copy back for reading.

`submit_ml.sh all` chains 2 → 6 (teacher `STAGE1_RUN`, default `s1_latent_cross`) — use it only once 1, 3 and 3b
have been checked by eye, with `DEBVADER_RUN` already the D1-trained run. Training jobs stop at 20 h (A100 limit) and resume from `runs/<name>/backup/` when resubmitted
with the same command.

## Overrides without editing the file

Every value of `ml_jeanzay.yaml` can be overridden: `dp2-ml $DP2_ML_CONFIG train-stage1 --ablation s1_latent_cross
--set stage1.latent_loss=gaussian_kl --set stage1.epochs=50 --run-name s1_kl_short`. A typo in a key is an error.

## Assumptions not verified yet (each one has a check)

| assumption | where it is checked |
|---|---|
| ~~paper_baseline reconstructs DP2 stamps~~ — refuted by phase 0 (noiseless training) | replaced by the D1-trained VAE |
| the D1-trained VAE reconstructs DP2 stamps down to the noise | phase 0 of the new run |
| MMU Euclid cutouts are centred on the object, north up, east left, 0.1″/px | D1 quality: astrometry, centroid offsets, e1/e2 correlation; `scale_ok` per cutout |
| MMU band names (`VIS`, `NIR_J`, ...) | probe |
| copaings position-angle convention | D1 quality: position-angle table |
| the Stage 1 acceptance thresholds | they are proposals; read the numbers |
