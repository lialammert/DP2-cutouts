<!--
README.md
Anna Preto
AstroDeep Project
Repository overview, installation and running the local pipeline.
-->

# DP2-cutouts

Code that builds a dataset of isolated-galaxy stamps: Rubin DP2 pixels (six bands, 59 × 59 pixels of
0.2 arcsec), Euclid Q1 catalogue (isolation, classification, redshift). The specification is
[dataset-recipe-short.md](dataset-recipe-short.md). The repository never contains data.

## Installation (macOS, Linux or WSL)

Python 3.11 or later (required by lsdb, hats, astropy and pyarrow). From the repository root:

```bash
python3 -m venv ~/venvs/dp2-cutouts
source ~/venvs/dp2-cutouts/bin/activate
pip install -e ".[notebook,test]"
pip install --no-deps "friendly @ git+https://github.com/LSSTDESC/friendly"
python -m ipykernel install --user --name dp2-cutouts --display-name "Python (dp2-cutouts)"
pytest tests
```

- `copaings` is installed from GitHub as a dependency.
- `friendly` is installed with `--no-deps` because its `pyproject` pins numpy < 2, which the rest does not need.
  Alternatively, clone it and point `FRIENDLY_PATH` (or `copaings.friendly_path` in the configuration) to it.
- If `python3 --version` is older than 3.11, install a recent Python (python.org or Homebrew on macOS) and
  create the environment with it, e.g. `python3.12 -m venv ~/venvs/dp2-cutouts`.
- `healpy` does not install on native Windows: use WSL.
- In Jupyter, choose the "Python (dp2-cutouts)" kernel.

Rubin platform token in `~/.rsp_token`, permissions 600, with the `read:tap` and `read:image` scopes.
The notebooks read the products of the pipeline: run `dp2-cutouts config/local_demo.yaml all` first.

## Running

```bash
dp2-cutouts config/local_demo.yaml all        # the whole pipeline
dp2-cutouts config/local_demo.yaml stamps     # a single step
pytest tests
```

| step | what it does | output (under `data_root`) |
|---|---|---|
| `rubin_catalog` | DP2 Object table through TAP, no cut | `10_rubin/catalog/object_box.parquet` |
| `euclid_catalog` | MER catalogue + redshift from HuggingFace (HATS), tiles in the copaings format | `00_euclid/` |
| `copaings` | copaings against Euclid; `prep` rewritten without the butler, the rest unchanged | `20_copaings/<name>/lsst_sb.parquet` |
| `rubin_images` | six SIA + SODA cutouts (image, mask, variance) of a region, patch bounding box | `10_rubin/images/` |
| `catalog_table` | one row per object, cuts 1, 2, 6, PSF per band, ellipse, redshift | `30_processed/catalog.parquet` |
| `stamps` | cuts 3, 4, 5, 7, 8, companion masking and the 0.02 decision | `30_processed/table.parquet`, `stamps.parquet` |
| `package` | final dataset in the MultimodalUniverse format, partitioned by order-5 HEALPix | `40_final/data/`, `40_final/catalog/` |

No row is removed before `package`: each cut writes its continuous quantity (`cutN_value`) and its condition
(`cutN_pass`). The cuts applied to the final dataset are listed in `package.apply`.

### Patch by patch (large areas)

Each patch is a case of the steps above, in its own folder `data_root/patches/<tract>_<patch>/`, with a sky box =
outer polygon of the patch + `patches.catalog_margin_arcsec`. A patch owns the objects whose `(tract, patch)` in the
Rubin Object table are its own; objects of the margin are context for copaings and masking only.

```bash
dp2-cutouts config/local_scale.yaml patches      # list of patches from SIA (no pixel), plan/patches.parquet
dp2-cutouts config/local_scale.yaml plan         # network: catalogues, copaings, cuts 1, 2, 6
dp2-cutouts config/local_scale.yaml fetch        # network: six full-patch images, only for patches with targets
dp2-cutouts config/local_scale.yaml process      # no network: cuts 3, 4, 5, 7, 8, masking
dp2-cutouts config/local_scale.yaml assemble     # final dataset of every processed patch -> data_root/final/
dp2-cutouts config/local_scale.yaml run          # the three phases, then assemble
dp2-cutouts config/local_scale.yaml invariance   # copaings on the whole area vs patch by patch
```

`--patches 5063_34 5063_35` restricts a phase to some patches, `--index N` selects the N-th patch (Slurm array),
`--force` reruns. `data_root/manifests/<tract>_<patch>.json` records, per patch and phase, the status, the
configuration fingerprint, counts, source identifiers and checksums, software versions and the last error. A rerun
skips what is already done with the same configuration; a change that matters to a phase reruns it and the later
ones. Data sources are chosen in the `sources` section (`adapters.py`): `tap`, `hf`, and `soda` or `butler` for the
pixels.

On Jean Zay, `scripts/jeanzay/submit.sh` submits plan and fetch on the `prepost` partition (internet access),
process on the compute nodes, and the assembly, as job arrays chained by dependencies.

### Non-regression

```bash
dp2-cutouts config/local_demo.yaml snapshot   # copy the current outputs to data_root/regression/
dp2-cutouts config/local_demo.yaml regress    # compare the current outputs with that copy, column by column
```

`tests/reference_fingerprints.json` holds two hashes of the reference demo (decisions and stamp pixels), which
reveal no DP2 data. `pytest tests` checks them whenever the demo products exist on the machine: it tells whether
another installation reproduces exactly the same dataset.

## Final dataset format

- `image`: list of structs, one per band `ugrizy`, `{band, flux, ivar, mask, scale}` as in
  MultimodalUniverse; `mask` is True on pixels not to be trusted (planes `package.mmu_mask_planes`).
- `image_maskbits`: the full Rubin mask planes, plus the `FILLED_SKY` bit (1 << 16) for pixels replaced by
  empty sky during companion masking.
- `40_final/catalog/`: the full catalogue before cuts, for reweighting.
- `has_valid_i_moments`: False when the Rubin object has no valid i-band ellipse. copaings then links it to its
  neighbours within 2" instead of testing ellipse overlaps, so its isolation rests on that distance only. These
  objects are kept; `dataset_info.json` gives the size of the strict selection that excludes them
  (`n_final_strict`).

## Notebooks

`notebooks/01_catalogs`, `02_cuts`, `03_masking`. Their outputs show DP2 pixels: clear them before committing.

## Probabilistic deblending campaign (Rubin × Euclid, Jean Zay)

Since 2026-11-24 the learning side compares three probabilistic deblenders of real Rubin blends on D1 / D2, with
Euclid as training-only information and weak lensing as the priority metric: package `src/dp2_cutouts/deblend/`,
commands `dp2-ml <cfg> deblend-stats | deblend-train | deblend-eval`, Slurm graph
`scripts/jeanzay/ml/submit_deblend.sh`. Design, data rules, run order and known limits:
[docs/deblend.md](docs/deblend.md).

## Cross-instrument learning, first architecture (archived)

On top of the final dataset: paired datasets (D1 isolated galaxies seen by both surveys, D2 Rubin blends that
Euclid resolves, controlled blends), Stage 1 (a Euclid encoder into the latent space of the frozen DebVAder VAE
`paper_baseline`) and Stage 2 (a Rubin blend encoder supervised by the Euclid teacher), with their evaluations.
Configuration `config/ml_jeanzay.yaml`, command `dp2-ml`, Slurm scripts `scripts/jeanzay/ml/`, notebooks
`notebooks/ml/`. Everything runs on Jean Zay; the order of the steps and what to check between them is in
[docs/cross_instrument.md](docs/cross_instrument.md).
