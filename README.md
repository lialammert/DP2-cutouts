<!--
README.md
Anna Preto
AstroDeep Project
Présentation du dépôt, installation et exécution de la chaîne locale.
-->

# rubin-euclid-stamps

Code de fabrication d'un jeu de vignettes de galaxies isolées : pixels Rubin DP2 (six bandes, 59 × 59 pixels de
0,2 seconde d'arc), catalogue Euclid Q1 (isolement, classification, redshift). La spécification est
[recette-nouveaux-datasets-abrege.md](recette-nouveaux-datasets-abrege.md). Le dépôt ne contient jamais de données.

## Installation (Linux ou WSL)

```bash
python3 -m venv ~/venvs/res
git clone https://github.com/LSSTDESC/copaings ~/src/copaings
git clone https://github.com/LSSTDESC/friendly ~/src/friendly
~/venvs/res/bin/pip install -e ~/src/copaings
~/venvs/res/bin/pip install -e ".[notebook,test]"
```

`friendly` n'est pas installé avec pip (son `pyproject` impose numpy < 2) : il est chargé par `FRIENDLY_PATH`,
donné dans la configuration (`copaings.friendly_path`). `healpy` ne s'installe pas sous Windows natif.

Jeton de la plateforme Rubin dans `~/.rsp_token`, droits 600, avec les portées `read:tap` et `read:image`.

## Exécution

```bash
res config/local_demo.yaml all        # toute la chaîne
res config/local_demo.yaml stamps     # une étape
pytest tests
```

| étape | ce qu'elle fait | sortie (sous `data_root`) |
|---|---|---|
| `rubin_catalog` | table Object de DP2 par TAP, sans coupure | `10_rubin/catalog/object_box.parquet` |
| `euclid_catalog` | catalogue MER + redshift depuis HuggingFace (HATS), tuiles au format copaings | `00_euclid/` |
| `copaings` | copaings contre Euclid ; `prep` réécrit sans butler, le reste inchangé | `20_copaings/<name>/lsst_sb.parquet` |
| `rubin_images` | six découpes SIA + SODA (image, masque, variance) d'une région, emprise du patch | `10_rubin/images/` |
| `catalog_table` | une ligne par objet, coupures 1, 2, 6, tache par bande, ellipse, redshift | `30_processed/catalog.parquet` |
| `stamps` | coupures 3, 4, 5, 7, 8, masquage des compagnes et décision à 0,02 | `30_processed/table.parquet`, `stamps.parquet` |
| `package` | jeu fini au format MultimodalUniverse, partitionné par HEALPix d'ordre 5 | `40_final/data/`, `40_final/catalog/` |

Aucune ligne n'est retirée avant `package` : chaque coupure écrit sa grandeur continue (`cutN_value`) et sa
condition (`cutN_pass`). Les coupures appliquées au jeu fini sont listées dans `package.apply`.

## Format du jeu fini

- `image` : liste de structures, une par bande `ugrizy`, `{band, flux, ivar, mask, scale}` comme
  MultimodalUniverse ; `mask` vaut True sur les pixels à ne pas croire (plans `package.mmu_mask_planes`).
- `image_maskbits` : les plans de masque Rubin complets, plus le bit `FILLED_SKY` (1 << 16) des pixels remplacés
  par du ciel vide au masquage des compagnes.
- `40_final/catalog/` : le catalogue complet avant coupures, pour la repondération.

## Notebooks

`notebooks/01_catalogues`, `02_coupures`, `03_masquage`. Leurs sorties montrent des pixels DP2. 
