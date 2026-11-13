import numpy as np

# KL incomplete-- only contains what's needed for phase 1 now

def fwhm_from_stamp(psf, pixel_scale, oversample=4):
    """
    Compute the FWHM in arcseconds of a PSF stamp.

    Parameters
    ----------
    psf : np.ndarray
        The PSF stamp, (P,P), assumed to be centered and normalized.
    pixel_scale : float
        The pixel scale in arcseconds/pixel.
    oversample : int, optional
        The oversampling factor for sub-pixel accuracy. Default is 4.

    Returns
    -------
    fwhm : float
        The FWHM of the PSF in arcseconds.
    """
    from scipy.ndimage import zoom

    # Oversample the PSF for better accuracy
    if oversample > 1:
        psf_oversampled = zoom(psf, oversample, order=3)
    else:
        psf_oversampled = psf

    area = np.count_nonzero(psf_oversampled >= 0.5 * np.max(psf_oversampled)) / (oversample ** 2)  # in px^2
    fwhm = 2.0 * np.sqrt(area / np.pi) * pixel_scale  # in arcseconds