# stamps.py
# Anna Preto
# AstroDeep Project
# Découpe des vignettes dans les six images d'une région, coupures sur pixels (3, 4, 5, 7, 8), masquage des
# compagnes avec remplissage par du ciel vide, et décision au masquage. Chaque coupure enregistre sa grandeur
# continue à côté de sa condition ; aucune cible n'est retirée ici.
#
# Deux gestes à ne pas rater :
# - la solution astrométrique rend (x, y) = (colonne, ligne), alors qu'un tableau s'indexe [ligne, colonne] ;
# - astropy rend des positions en indices 0 (le premier pixel FITS vaut 1) : on n'ajoute ni ne retire rien.
# Le centre entier d'une vignette est floor(x + 0,5), la position fractionnaire est gardée pour les mesures.

import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, label

from ..common import masks
from ..common.ellipse import (FWHM2SIGMA, euclid_ellipse, moments_to_ellipse, psf_ellipse,
                              radius_map)
from ..common.shapes import adaptive_moments, circular_weight, delta_e
from ..sources.rubin_soda import BANDS, tract_origin
from .catalog_table import copaings_dir

NJY_ZEROPOINT = 31.4  # magnitude AB d'un flux de 1 nJy


# ----------------------------------------------------------------------------- images
def load_images(cfg):
    img_dir = cfg.data("10_rubin", "images", "x").parent
    meta = json.load(open(img_dir / "images.json"))
    flux, var, mask, wcs, origins, bits = [], [], [], [], [], []
    for b in BANDS:
        with fits.open(img_dir / meta["files"][b]) as h:
            flux.append(h["IMAGE"].data.astype(np.float32))
            var.append(h["VARIANCE"].data.astype(np.float32))
            mask.append(h["MASK"].data.astype(np.int32))
            wcs.append(WCS(h["IMAGE"].header))
            origins.append(tract_origin(h["IMAGE"].header))
            bits.append(masks.plane_bits(h["MASK"].header))
    if len({f.shape for f in flux}) != 1 or len(set(origins)) != 1 or any(bt != bits[0] for bt in bits):
        raise RuntimeError("les six images n'ont pas la même grille de pixels ou les mêmes plans de masque")
    mask = np.stack(mask)
    # Pixels détectés dans au moins une bande : une compagne peut n'être détectée que dans certaines couleurs.
    detected = np.any((mask & masks.bitmask(bits[0], ["DETECTED"])) != 0, axis=0)
    return dict(flux=np.stack(flux), var=np.stack(var), mask=mask, detected=detected, wcs=wcs, x0=origins[0][0],
                y0=origins[0][1], bits=bits[0], meta=meta)


# ----------------------------------------------------------------------------- compagnes
def companions(cfg, img, cat):
    # Tous les objets connus dans la région, sans aucune coupure : Rubin (y compris ponctuels et sans
    # i_extendedness) et Euclid (y compris étoiles), chacun avec son ellipse dans les pixels de la région.
    w = img["wcs"][BANDS.index("i")]
    raw = pd.read_parquet(cfg.data("10_rubin", "catalog", "object_box.parquet"))
    a, b, th = moments_to_ellipse(raw["i_ixx"], raw["i_iyy"], raw["i_ixy"])
    ap, bp, thp = psf_ellipse(raw["i_ixxPSF"], raw["i_iyyPSF"])
    bad = ~(np.isfinite(a) & np.isfinite(b) & (b > 0))
    a, b, th = np.where(bad, ap, a), np.where(bad, bp, b), np.where(bad, thp, th)
    x, y = w.world_to_pixel_values(raw["coord_ra"].values, raw["coord_dec"].values)
    rubin = pd.DataFrame(dict(survey="rubin", id=raw["objectId"].astype("int64"), x=x, y=y, a=a, b=b, theta=th))

    # Ellipses Euclid : même construction que le test de recouvrement de copaings (catalogs.build_euclid puis
    # true_params), relue dans les tuiles qu'a lues copaings, donc avec la même convention d'angle.
    from .run_copaings import _import_copaings, make_case
    catalogs, _, _ = _import_copaings(cfg)
    case = make_case(cfg)
    lsst_cop = pd.read_parquet(copaings_dir(cfg) / "catalogs" / "lsst.parquet")
    eu = catalogs.build_euclid(case, lsst_cop)
    # Tache : médiane de la tache Rubin en bande i sur la région (copaings prend celle du premier membre du groupe).
    fwhm = float(np.nanmedian(np.asarray(cat["psf_sigma_i"], float))) * FWHM2SIGMA
    ea, eb, eth = euclid_ellipse(eu["size_true"], eu["size_minor_true"], eu["position_angle_true"], fwhm)
    ex, ey = w.world_to_pixel_values(eu["ra"].values, eu["dec"].values)
    euclid = pd.DataFrame(dict(survey="euclid", ra=eu["ra"].values, dec=eu["dec"].values, x=ex, y=ey,
                               a=ea, b=eb, theta=eth))
    ok = np.isfinite(euclid["a"]) & np.isfinite(euclid["b"])
    euclid = euclid[ok].reset_index(drop=True)
    return rubin, euclid


def companion_mask(shape, ox, oy, comp, pad):
    # Méthode « ellipse » de la recette : union des ellipses de demi-lumière des compagnes, boîte d'origine (ox, oy).
    h, w = shape
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    m = np.zeros(shape, bool)
    for r in comp[sel].itertuples():
        m |= radius_map(shape, r.x - ox, r.y - oy, r.a, r.b, r.theta) <= 1.0
    return m


def companion_mask_footprint(detected, shape, ox, oy, target, comp, pad, grow_px, shared_k):
    # Méthode « footprint » (écart à la recette, validé) : l'ellipse de demi-lumière laissait les ailes des compagnes
    # brillantes autour de trous remplis de ciel. On s'appuie sur les empreintes de détection (plan DETECTED, dans au
    # moins une bande, composantes connexes) :
    # - l'empreinte qui contient la cible lui appartient : on n'y touche pas, ses ailes restent entières ;
    # - toute autre empreinte de la boîte est une autre source : masquée en entier, élargie de grow_px pixels sans
    #   mordre sur l'empreinte de la cible ;
    # - une compagne dont le centre tombe dans l'empreinte de la cible (empreinte partagée) : shared_k fois son
    #   ellipse, à l'intérieur de cette empreinte ;
    # - une compagne non détectée (vue par Euclid seul) : son ellipse de demi-lumière.
    h, w = shape
    det = detected[oy:oy + h, ox:ox + w]
    labels, _ = label(det, structure=np.ones((3, 3), int))
    xt, yt = target[0], target[1]
    it, jt = int(np.floor(yt + 0.5)), int(np.floor(xt + 0.5))
    lt = labels[it, jt] if (0 <= it < h and 0 <= jt < w) else 0
    target_fp = labels == lt if lt > 0 else np.zeros(shape, bool)

    m = det & ~target_fp
    if grow_px > 0 and m.any():
        m = binary_dilation(m, iterations=int(grow_px)) & ~target_fp
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    for r in comp[sel].itertuples():
        x, y = r.x - ox, r.y - oy
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        inside = 0 <= ix < w and 0 <= iy < h
        if inside and target_fp[iy, ix]:
            m |= (radius_map(shape, x, y, r.a, r.b, r.theta) <= shared_k) & target_fp
        elif not (inside and det[iy, ix]):
            m |= radius_map(shape, x, y, r.a, r.b, r.theta) <= 1.0
    return m


# ----------------------------------------------------------------------------- ciel vide
class SkyBank:
    # Fenêtres de ciel de la région où aucun pixel n'est détecté, abîmé ou non exposé, dans aucune bande.
    def __init__(self, img, unusable_mask, size, stride=4):
        self.unusable = np.any((img["mask"] & unusable_mask) != 0, axis=0)
        H, W = self.unusable.shape
        rr, cc = np.meshgrid(np.arange(0, H - size + 1, stride), np.arange(0, W - size + 1, stride), indexing="ij")
        self.corners = np.column_stack([rr.ravel(), cc.ravel()])
        self.size = size

    def find(self, need, rng, tries):
        # Fenêtre dont les pixels utilisables couvrent tous les pixels à remplir ; à défaut, la moins mauvaise.
        n = self.size
        order = rng.permutation(len(self.corners))[:tries]
        best, best_bad = None, np.inf
        for k in order:
            r, c = self.corners[k]
            nbad = int(np.count_nonzero(self.unusable[r:r + n, c:c + n] & need))
            if nbad < best_bad:
                best, best_bad = (r, c), nbad
                if nbad == 0:
                    break
        return best, best_bad


def fill(img, ox, oy, size, need, window):
    # Copie flux, variance et bits du ciel choisi sur les pixels à remplir, et lève le bit FILLED_SKY.
    sl = np.s_[:, oy:oy + size, ox:ox + size]
    flux, var, mbits = img["flux"][sl].copy(), img["var"][sl].copy(), img["mask"][sl].copy()
    if need.any():
        r, c = window
        ws = np.s_[:, r:r + size, c:c + size]
        flux[:, need] = img["flux"][ws][:, need]
        var[:, need] = img["var"][ws][:, need]
        mbits[:, need] = img["mask"][ws][:, need] | masks.FILLED_BIT
    return flux, var, mbits


# ----------------------------------------------------------------------------- une cible
def process_target(row, img, cfg, rubin, euclid, banks, rng):
    cc = cfg["cuts"]
    half, dhalf = cc["stamp_px"] // 2, cc["diag_px"] // 2
    H, W = img["flux"].shape[1:]
    bits = img["bits"]
    nodata = masks.bitmask(bits, ["NO_DATA"])
    bad = masks.bitmask(bits, cc["bad_planes"])
    out = dict(object_id=row.object_id)

    # Positions dans chaque bande ; coupure 7 : même pixel entier partout.
    pos = np.array([w.world_to_pixel_values(row.ra, row.dec) for w in img["wcs"]], float)
    ib = BANDS.index(cc["shape_band"])
    x, y = pos[BANDS.index("i")]
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    centers = np.floor(pos + 0.5).astype(int)
    out["x_tract"], out["y_tract"] = img["x0"] + x, img["y0"] + y
    out["cut7_value"] = float(np.max(np.hypot(*(pos - pos[BANDS.index("i")]).T)))
    out["cut7_pass"] = bool(np.all(centers == centers[BANDS.index("i")]))

    # Démo locale : la vignette doit tenir dans la région téléchargée (sur la plateforme, c'est le patch entier).
    out["in_region"] = bool(half <= ix < W - half and half <= iy < H - half)
    if not out["in_region"]:
        return out, None

    # Coupure 3 : marge entière dans l'emprise du patch, repère du tract.
    pb = img["meta"]["patch_bbox"]
    tx, ty = img["x0"] + ix, img["y0"] + iy
    margin = min(tx - pb["x_min"], pb["x_max"] - tx, ty - pb["y_min"], pb["y_max"] - ty)
    out["cut3_value"] = int(margin)
    out["cut3_pass"] = bool(margin >= cc["margin_px"])

    n = cc["stamp_px"]
    ox, oy = ix - half, iy - half
    xc, yc = x - ox, y - oy  # position fractionnaire dans la vignette (colonne, ligne)
    box = np.s_[:, oy:oy + n, ox:ox + n]
    mbox = img["mask"][box]

    # Coupure 4 : aucun pixel non exposé dans la boîte, dans aucune bande.
    nod = (mbox & nodata) != 0
    out["cut4_value"] = int(nod.sum())
    out["cut4_pass"] = out["cut4_value"] == 0

    # Coupure 5 : rayon réduit du pixel abîmé le plus proche, dans l'ellipse de la cible.
    a, b, th = row.ellipse_a, row.ellipse_b, row.ellipse_theta
    if not (np.isfinite(a) and np.isfinite(b) and b > 0):
        a = b = float(np.sqrt(2 * np.log(2)) * row.psf_sigma_i / 0.2)
        th = 0.0
    rmap = radius_map((n, n), xc, yc, a, b, th)
    badpix = np.any((mbox & bad) != 0, axis=0)
    out["cut5_value"] = float(rmap[badpix].min()) if badpix.any() else np.inf
    out["cut5_n_bad_in_k"] = int(np.count_nonzero(badpix & (rmap <= cc["k_ellipse"])))
    out["cut5_pass"] = bool(out["cut5_value"] > cc["k_ellipse"])

    # Compagnes : tout objet de la boîte hors du groupe de la cible (la cible Rubin et sa contrepartie Euclid).
    rc = rubin[rubin["id"] != row.object_id]
    ec = euclid
    if pd.notna(row.euclid_ra):
        ec = euclid[~((euclid["ra"] == row.euclid_ra) & (euclid["dec"] == row.euclid_dec))]
    comp = pd.concat([rc[["x", "y", "a", "b", "theta"]], ec[["x", "y", "a", "b", "theta"]]], ignore_index=True)
    pad = float(np.nanmax(comp["a"])) if len(comp) else 0.0
    pad = min(pad, 50.0)
    mc = cfg["masking"]

    def make_need(size, bx, by):
        if mc["method"] == "ellipse":
            return companion_mask((size, size), bx, by, comp, pad)
        if mc["method"] == "footprint":
            return companion_mask_footprint(img["detected"], (size, size), bx, by, (x - bx, y - by, a, b, th),
                                            comp, pad, mc["grow_px"], mc["shared_k"])
        raise ValueError("masking.method doit valoir « ellipse » ou « footprint »")

    need = make_need(n, ox, oy)
    out["masked_fraction"] = float(need.mean())
    out["masked_fraction_in_ellipse"] = float(need[rmap <= 1].mean()) if (rmap <= 1).any() else np.nan
    out["n_companions"] = int(((comp["x"] - ox + 0.5).between(0, n) & (comp["y"] - oy + 0.5).between(0, n)).sum())

    window, nbad = banks[n].find(need, rng, cc["sky_tries"]) if need.any() else (None, 0)
    flux, var, mbits = fill(img, ox, oy, n, need, window)
    out["sky_fill_bad_frac"] = float(nbad / need.sum()) if need.any() else 0.0
    if window is not None:
        out["sky_window_x_tract"], out["sky_window_y_tract"] = img["x0"] + window[1], img["y0"] + window[0]

    # Décision au masquage : écart d'ellipticité avant / après, en bande de forme.
    weight = (~np.any(nod, axis=0)).astype(np.int32)
    m0 = adaptive_moments(img["flux"][box][ib], weight, (xc, yc))
    m1 = adaptive_moments(flux[ib], weight, (xc, yc))
    out.update(e1_before=m0["e1"], e2_before=m0["e2"], e1=m1["e1"], e2=m1["e2"], sigma_hsm=m1["sigma"],
               shape_ok=bool(m0["ok"] and m1["ok"]))
    out["delta_e_mask"] = delta_e(m0, m1) if out["shape_ok"] else np.nan
    # La mesure de forme ne sélectionne jamais : une mesure ratée ne fait pas écarter la vignette.
    out["mask_pass"] = bool(not np.isfinite(out["delta_e_mask"]) or out["delta_e_mask"] <= cc["delta_e_max"])

    # Profondeur locale : variance médiane du ciel (pixels ni détectés ni non exposés), magnitude limite à 5 sigma
    # pour une source ponctuelle de la taille de la tache.
    det = masks.bitmask(bits, ["DETECTED", "NO_DATA"])
    for k, band in enumerate(BANDS):
        sky = (mbox[k] & det) == 0
        v = float(np.median(img["var"][box][k][sky])) if sky.any() else np.nan
        s_pix = float(getattr(row, f"psf_sigma_{band}")) / 0.2
        out[f"sky_var_{band}"] = v
        out[f"depth5_{band}"] = NJY_ZEROPOINT - 2.5 * np.log10(5.0 * np.sqrt(4 * np.pi * s_pix ** 2 * v))

    # Coupure 8 : vignette de diagnostic deux fois plus grande, masquée de la même façon, forme mesurée dans deux
    # ouvertures circulaires.
    nd = cc["diag_px"]
    dox, doy = ix - dhalf, iy - dhalf
    diag = None
    out["cut8_evaluated"] = False
    if 0 <= dox and dox + nd <= W and 0 <= doy and doy + nd <= H:
        dneed = make_need(nd, dox, doy)
        dwin, dbad = banks[nd].find(dneed, rng, cc["sky_tries"]) if dneed.any() else (None, 0)
        dflux, _, dbits = fill(img, dox, doy, nd, dneed, dwin)
        dxc, dyc = x - dox, y - doy
        dok = ((dbits[ib] & nodata) == 0)
        r_in, r_out = cc["aperture_radii"]
        ma = adaptive_moments(dflux[ib], circular_weight((nd, nd), dxc, dyc, r_in) * dok, (dxc, dyc))
        mb = adaptive_moments(dflux[ib], circular_weight((nd, nd), dxc, dyc, r_out) * dok, (dxc, dyc))
        out["cut8_value"] = delta_e(ma, mb) if (ma["ok"] and mb["ok"]) else np.nan
        out["cut8_evaluated"] = bool(np.isfinite(out["cut8_value"]))
        out["diag_sky_fill_bad_frac"] = float(dbad / dneed.sum()) if dneed.any() else 0.0
        diag = dict(diag_flux=dflux[ib].astype(np.float32).ravel(), diag_need=dneed.ravel())
    out["cut8_pass"] = bool(not out["cut8_evaluated"] or out["cut8_value"] <= cc["delta_e_max"])

    with np.errstate(divide="ignore"):
        ivar = np.where((var > 0) & ((mbits & nodata) == 0), 1.0 / var, 0.0).astype(np.float32)
    arrays = dict(object_id=row.object_id,
                  flux_raw=img["flux"][box].astype(np.float32).ravel(),
                  flux=flux.astype(np.float32).ravel(), ivar=ivar.ravel(), maskbits=mbits.astype(np.int32).ravel(),
                  companion_mask=need.ravel())
    if diag:
        arrays.update(diag)
    return out, arrays


def run(cfg):
    cc = cfg["cuts"]
    img = load_images(cfg)
    cat = pd.read_parquet(cfg.data("30_processed", "catalog.parquet"))
    meta = img["meta"]
    targets = cat[cat["cut1_pass"] & cat["cut2_pass"] & (cat["tract"] == meta["tract"])
                  & (cat["patch"] == meta["patch"])]
    rubin, euclid = companions(cfg, img, cat)

    unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cc["bad_planes"]))
    banks = {n: SkyBank(img, unusable, n) for n in (cc["stamp_px"], cc["diag_px"])}
    rows, arrays = [], []
    for row in targets.itertuples(index=False):
        rng = np.random.default_rng([int(cc["seed"]), int(row.object_id) & 0x7FFFFFFF])
        out, arr = process_target(row, img, cfg, rubin, euclid, banks, rng)
        rows.append(out)
        if arr is not None:
            arrays.append(arr)
    res = pd.DataFrame(rows)

    # Table complète : toutes les lignes du catalogue, colonnes de pixels vides hors cibles.
    table = cat.merge(res, on="object_id", how="left")
    table["pixel_target"] = table["object_id"].isin(res["object_id"])
    table["in_region"] = table["in_region"].astype("boolean").fillna(False).astype(bool)
    table.to_parquet(cfg.data("30_processed", "table.parquet"), index=False)

    st = pd.DataFrame(arrays)
    st.attrs = {}
    st.to_parquet(cfg.data("30_processed", "stamps.parquet"), index=False)
    with open(cfg.data("30_processed", "stamps.json"), "w") as f:
        json.dump(dict(bands=list(BANDS), stamp_px=cc["stamp_px"], diag_px=cc["diag_px"], plane_bits=img["bits"],
                       filled_bit=masks.FILLED_BIT, tract=meta["tract"], patch=meta["patch"]), f, indent=2)

    t = table[table["in_region"]]
    msg = {k: int(t[f"{k}_pass"].sum()) for k in ("cut3", "cut4", "cut5", "cut6", "cut7", "cut8", "mask")}
    print(f"cibles du patch {len(targets):,} ; dans la région {len(t):,} ; passent : {msg}")
    return table
