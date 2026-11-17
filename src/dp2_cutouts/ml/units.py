# units.py
# Anna Preto
# AstroDeep Project
# Flux units between DP2 (nJy) and the units DebVAder paper_baseline was trained in (BTK electrons). BTK renders a
# galaxy of AB magnitude m into exposure_time * 10**(-0.4 (m - zeropoint)) electrons, with the survey parameters of
# surveycodex; a flux of 1 nJy is AB magnitude 31.4. The factor per band is therefore the electron count of a 31.4
# magnitude source, read from surveycodex itself rather than copied here.

import numpy as np

NJY_AB_MAG = 31.4


def lsst_electrons_per_njy(bands, survey_name="LSST"):
    import surveycodex
    from surveycodex.utilities import mag2counts

    survey = surveycodex.get_survey(survey_name)
    out = []
    for b in bands:
        filt = survey.get_filter(b)
        try:
            counts = mag2counts(NJY_AB_MAG, survey, filt)
        except (TypeError, AttributeError):
            counts = mag2counts(NJY_AB_MAG, survey_name, b)
        value = counts.to_value("electron") if hasattr(counts, "to_value") else float(counts)
        out.append(float(value))
    out = np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(out) & (out > 0)):
        raise ValueError(f"invalid nJy -> electron factors from surveycodex: {out}")
    return out


class UnitConverter:
    # flux_vae = factor * flux_nJy, ivar_vae = ivar_nJy / factor**2, per band (last axis).
    def __init__(self, mode, bands):
        self.mode = mode
        self.bands = list(bands)
        if mode == "nJy":
            self.factor = np.ones(len(bands))
        elif mode == "lsst_electrons":
            self.factor = lsst_electrons_per_njy(bands)
        else:
            raise ValueError(f"unknown unit mode {mode!r} (nJy | lsst_electrons)")

    def flux(self, flux_njy):
        return (np.asarray(flux_njy, np.float64) * self.factor).astype(np.float32)

    def ivar(self, ivar_njy):
        return (np.asarray(ivar_njy, np.float64) / self.factor ** 2).astype(np.float32)

    def flux_back(self, flux_vae):
        return (np.asarray(flux_vae, np.float64) / self.factor).astype(np.float32)

    def describe(self):
        return dict(mode=self.mode, bands=self.bands, factor=[float(v) for v in self.factor])

    @classmethod
    def for_vae(cls, cfg, vae):
        # rubin.vae_units "auto" takes the units of the data the VAE was trained on (vae.units_mode: nJy for a
        # real-data build, lsst_electrons for the BTK simulations). An explicit value must agree with it: feeding
        # a VAE other units than its training set's is exactly the factor-of-fifty error resolve_stats_path warns of.
        mode = cfg["rubin"]["vae_units"]
        if mode == "auto":
            mode = vae.units_mode
        elif mode != vae.units_mode:
            raise ValueError(f"rubin.vae_units={mode!r} but the VAE of {vae.run_dir} was trained in "
                             f"{vae.units_mode!r}; use 'auto'")
        return cls(mode, cfg["rubin"]["bands"])
