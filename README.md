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

## Installation (Linux or WSL)

```bash
python3 -m venv ~/venvs/dp2-cutouts
git clone https://github.com/LSSTDESC/copaings ~/src/copaings
git clone https://github.com/LSSTDESC/friendly ~/src/friendly
~/venvs/dp2-cutouts/bin/pip install -e ~/src/copaings
~/venvs/dp2-cutouts/bin/pip install -e ".[notebook,test]"
```

`friendly` is not installed with pip (its `pyproject` pins numpy < 2): it is loaded through `FRIENDLY_PATH`,
given in the configuration (`copaings.friendly_path`). `healpy` does not install on native Windows.

Rubin platform token in `~/.rsp_token`, permissions 600, with the `read:tap` and `read:image` scopes.

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

## Final dataset format

- `image`: list of structs, one per band `ugrizy`, `{band, flux, ivar, mask, scale}` as in
  MultimodalUniverse; `mask` is True on pixels not to be trusted (planes `package.mmu_mask_planes`).
- `image_maskbits`: the full Rubin mask planes, plus the `FILLED_SKY` bit (1 << 16) for pixels replaced by
  empty sky during companion masking.
- `40_final/catalog/`: the full catalogue before cuts, for reweighting.

## Notebooks

`notebooks/01_catalogs`, `02_cuts`, `03_masking`. Their outputs show DP2 pixels: clear them before committing.
