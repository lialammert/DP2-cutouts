<!--
deblend.md
Anna Preto
AstroDeep Project
Design, data rules, run order and known limits of the probabilistic deblending campaign (Runs A, B, C).
-->

# Probabilistic deblending of real Rubin blends — three architectures (Jean Zay)

Started 2026-11-24. It replaces the Stage 1 / Stage 2 architecture (Euclid encoder into the frozen DebVAder
latent, then a blend student). That architecture is archived: its code stays runnable for traceability
(`train-stage1`, `eval-stage1`, `train-stage2`, `eval-stage2`, `phase0`), and its runs are moved to `$STORE` (see
below). It is no longer the active research pipeline. The dataset pipeline (stamps, D1, D2) is unchanged. The one
addition is an optional D1 array, described below.

**Goal.** Reconstruct the central Rubin galaxy of a real blend. The output is deblended and denoised, in the six
bands ugrizy, still convolved by the Rubin PSF, and comes as a predictive distribution. At inference only Rubin
is available. Euclid is privileged information, used in training and in some evaluations. Models are compared
for weak lensing first: shapes, sizes, selection, and the shear response where it can be measured.

**Hard rules.**
- No synthetic blends, for training or for this evaluation campaign.
- No BTK/GalSim supervision.
- No reconstruction of the neighbours.
- No PSF matching or deconvolution in the networks.
- No transformer, attention, PoE or MoE.
- No InfoNCE.
- No explicit ellipticity loss.
- VIS only until the NIR flux scale is established.
- The holdout split is never used for any choice.

## The three runs

| | teacher (training only) | student (inference: Rubin blend X only) | decoder |
|---|---|---|---|
| A | K1: Euclid VIS VAE, q_E(z \| E_c, M_E), frozen | CNN + MC dropout 0.1, latent MSE to sampled z_E | K2 (Rubin from z_E), frozen |
| B | K1, same checkpoint | CNN context h(X) 128-D + CFM field (41 376 parameters) | K2, same checkpoint |
| C | C2: r(z \| X, E_c), full covariance, on the C1 joint latent | CNN context + the same CFM field (C3), D1 replay | C1 Rubin decoder, frozen |

**Stages.**
- `rubin_vae` is the classical Rubin VAE: Rubin in, Rubin out. It is the reference for the D1 comparisons.
- `k1` and `k2` are shared by Runs A and B: one checkpoint each, frozen after training.
- `c1` is the joint VAE: Rubin and Euclid embedders, a fusion MLP, a diagonal Gaussian latent, and two decoders.
  Its loss is a proper ELBO at β = 1.
- `c2` is the conditional teacher r(z | X, E_c) together with a Gaussian student s(z | X). The two share one Rubin
  backbone, initialised from a copy of C1's.
  - Loss on D2: E_r[−log p_E(E_c | z)] + KL(r ‖ s).
  - Plus λ_D1 [½ KL(q_A ‖ s(R)) + ½ KL(q_A ‖ r(R, E))] on D1. Here q_A comes from the frozen C1, and D1 and D2
    batches are balanced.
  - λ_D1 = `auto` is the median ratio of the two gradient norms on the shared backbone, measured at
    initialisation (JAX) and then kept fixed.
- `c3` is a CFM student on samples of the frozen r, with a D1 replay on samples of q_A. Its context is
  initialised from C2's Rubin backbone.

**Likelihoods.**
- Every image likelihood is an observational Gaussian in physical units: Rubin in nJy, VIS as delivered by MER
  (µJy per pixel).
- It is summed over the pixels that are observations of the target, with the stored inverse variance.
- The reconstruction terms of the two instruments and the KL terms are therefore on the same footing; no
  arbitrary pixel weight is used (`weight_r`, `weight_e` = 1).
- Decoders output the expected signal (no noise) in units of a per-band scale s_b, multiplied back before the
  likelihood.
- The coadd noise is spatially correlated; the per-pixel inverse variance ignores this, and a χ² near 1 is not
  taken as proof of correct denoising.

**CFM.**
- Linear path z_t = (1 − t) z0 + t z1, with target velocity u = z1 − z0 and loss mean ‖v − u‖².
- No ODE is solved during training.
- Sampling uses fixed-step Euler or Heun (NFE = 2 × steps for Heun). The context is computed once per blend and
  shared by all steps and samples.
- Latents are standardised with statistics from the train split (teacher samples) and mapped back before
  decoding.

## Data rules

**Inputs.**
- Every input is tanh(arcsinh(2.5 f / s_b)), with s_b computed on the D1 train split (`deblend-stats`).
- The Euclid input has a second channel, the validity of each pixel. The K1 / C1 / C2 encoders are therefore
  mask-aware, and a masked neighbour is never taken for sky.

**Pixels that are not observations of the target.** They are replaced in the input by a noise draw at their own
inverse variance, and excluded from the likelihoods:
- Rubin: pixels with an MMU mask plane set (NO_DATA, SATURATED, CR, INTERPOLATED, DETECTION_EDGE) or with
  ivar ≤ 0.
- Euclid: pixels with an MMU mask set, with ivar ≤ 0, or belonging to another source (`euclid_fill`).

**Rubin pixels filled with sky by the stamp pipeline.** They are real sky pixels from the same region, so they
count as observations (decision of 2026-11-24).

**D1 Euclid neighbours.**
- `dataset1.euclid_neighbour_mask: segmentation` flags the other sources on each D1 Euclid cutout, with the D2
  rules: MER segments, or ellipses where no tile covers the cutout. The result is stored in the array
  `euclid_fill`; the pixels themselves are not changed.
- This puts the D1 Euclid inputs on the same footing as the masked central E_c of D2.
- D1 must be built with this option. The reader refuses a D1 without it unless
  `deblend.require_d1_euclid_fill: false`.

**D2 centring.**
- The Rubin input is the blend stamp centred on the Rubin object of the group (`rubin_*_rc`), which is what a
  Rubin-only user has.
- The target is the Euclid member of the group nearest to that centre: one target per group, so that one input
  never has two targets.
- The stamps centred on the Euclid target (`d2_centering: euclid`) carry Euclid information that is not
  available at inference; they are kept for diagnostics only.

**Splits.**
- The patch hash is shared by D1 and D2 (70 / 10 / 10 / 10): no patch feeds two splits.
- Checkpoints are selected on the validation loss only.
- The holdout split is read only by `submit_deblend.sh final`.

## Order on Jean Zay

1. Stamp pipeline assembled.
2. Build D1 (with `euclid_neighbour_mask`) and D2: `submit_ml.sh datasets`. The controlled blends it also
   builds are not used.
3. `submit_deblend.sh pilot`: every stage for 300 steps (run suffix `_pilot`), then every evaluation on
   500 val objects. Check from it:
   - finite losses and the training speed;
   - the logs of `logs/deblend_*`;
   - `frozen_check.json` of each run;
   - `run.json` (λ_D1 calibration, parameter counts).

   Then set `deblend.train.train_steps` from the measured speed.
4. `submit_deblend.sh all`: the eight trainings as chains of 2-hour dev-QoS jobs that resume from their backups,
   then the evaluations on the test split.
5. `SEEDS="1 2" submit_deblend.sh seeds`: replicates of the students (A, B, C2, C3).
6. `submit_deblend.sh final`: holdout, once.

**Outputs** go under `$DP2_ML_ROOT/deblend/`:
- `norm_stats.json`;
- `runs/<name>/`: `run.json`, the config snapshot, `history.csv`, the weights, `backup/`,
  `frozen_check.json`, `latent_stats.json`, `DONE`;
- `evaluations/<what>_<split>/`: `objects.parquet`, `summary.json`, figures.

**Archiving the old architecture:**

```bash
mkdir -p $STORE/dp2-archive && tar -C $DP2_ML_ROOT -cf $STORE/dp2-archive/legacy_stage12_runs.tar runs evaluations
```

Run it only if those folders exist; the git tag `legacy-stage12` marks the code.

## Evaluations (`deblend-eval`)

| what | content |
|---|---|
| `d1` | Isolated galaxies. What each latent restores in Rubin: R→R (Rubin VAE), E→R (K1 + K2), (R, E)→R (C1). Then the three final models applied to isolated Rubin stamps. Measured on the i band: shapes (adaptive moments, and REGAUSS with a Gaussian PSF from the catalogue PSF moments), size, aperture fluxes in six bands, colours, HSM failures. Observational predictive calibration of the samples (a noise draw at the observed ivar, then measured): coverage at 50 / 68 / 90 / 95 %, PIT, CRPS, energy score. |
| `masking` | The K1 latent on D1 Euclid stamps, with and without real D2 neighbour masks: latent shift in posterior σ, and the change in decoded flux, size and shape, by mask fraction. |
| `d2` | Real blends, the same objects for all runs. Shapes are measured both as the mean of the per-sample shapes and as the shape of the mean image. Euclid VIS REGAUSS shapes are read through the Rubin-vs-VIS relation fitted on D1 val. Also reported: \|e\| distribution, position angle, alignment with the neighbour, size, over-subtraction of the blend, HSM failures. Results are binned by blendedness, S/N, magnitude, separation, contrast, resolution, mask severity and field. Paired run-to-run differences come with a patch block bootstrap. |
| `cost` | Parameters; time per galaxy for 10 / 50 / 100 samples; context time; CFM latent convergence against a 200-step Heun reference for Euler and Heun at NFE 5 to 100; peak GPU memory. |
| `latent` | K1 and C1 latents: variance per dimension, PCA spectrum, participation ratio. C1 decoder Jacobians, noise-weighted: the directions that move the Rubin image but hardly the Euclid one, and their effect on Rubin fluxes, size and shape. |

## Known limits, stated rather than hidden

- **No isolated Rubin truth on D2.** D2 numbers are comparisons with references that are themselves
  measurements, never a calibration of the Rubin posterior.
- **The D2 reference is Euclid VIS read through a D1 relation.** That relation may not transfer without bias to
  blended targets.
- **REGAUSS uses a Gaussian PSF built from catalogue moments.** The datasets do not store the PSF images; a
  shear-response test of the metacalibration type would need them, or the same Gaussian approximation. The
  shear response is therefore not measured yet: see the open items.
- **MC dropout diversity is not a calibrated uncertainty by construction.** The latent MSE penalises the
  student's variance: E‖Z_B − Z_E‖² = ‖E Z_B − E Z_E‖² + tr Cov Z_B + tr Cov Z_E.
- **The CFM of Run B learns the mixture of K1 posteriors over the E_c compatible with X.** It is bounded by the
  Euclid-only teacher.
- **Run C's teacher bounds Run C's student.** The C2 objective is a variational bound of p(E_c | X): it does not
  constrain the Rubin information to which Euclid is insensitive. The `latent` Jacobian analysis looks for that
  information.
- **The comparison is between complete pipelines, not a single-variable ablation.** Run C uses D1 during its
  D2 stages; Runs A and B do not.
- **The Grishin et al. manuscript could not be retrieved.** It is not on arXiv, the web or GitHub as of
  2026-11-24. K1 / K2 / A follow the campaign specification: latent MSE (squared L2 norm averaged over
  dimensions and batch), whereas the manuscript is reported to write an L2 norm without the square.
