---
title: Short recipe - real dataset, isolated galaxies
---

# Short recipe

Short version of the full recipe (`recette-nouveaux-datasets.md`, in French), which carries the justifications.
Here, one sentence per step and one sentence per limitation.

## Before starting

| need | why |
|---|---|
| an account on a computing centre | cut and measure the stamps |
| an account on the Rubin platform, with the processing pipelines and the butler | the `copaings` preparation step needs it |
| a second machine with open internet access | the Euclid catalogue is often blocked by computing-centre firewalls |
| an access token for the Rubin platform, in `~/.rsp_token`, permissions 600 | download the images; never print it or put it under version control |
| `copaings` and `friendly` | matching, ellipse overlaps and blending entropy |
| `lsdb`, `pyvo`, `galsim`, `numpy`, `scipy`, `astropy` | read the catalogues and measure shapes |

**The two sources.** Rubin provides the pixels: six deep images per sky area, one per colour, 0.2 arcsec per
pixel, stamps of 59 × 59 pixels. Euclid provides the catalogue: it sees twice as sharply, says whether a source is
a galaxy, and gives the redshift.

**The rule.** Euclid certifies isolation. Rubin can only veto: it is never asked to guarantee that there is no
neighbour, since it would answer "nothing" wherever it cannot tell the neighbour apart.

**Step 1.** Download the Euclid catalogue area by area, cumulatively, until the desired number of galaxies is
reached, never removing an area already drawn. The goal is a number of galaxies at the end of the pipeline, not a
number of areas.

## The eight cuts, in order

The first two read only the catalogues: they are applied **before any image download**. The following ones work
on the pixels.

| # | what we do | accepted limitation |
|---|---|---|
| 1 | Keep the sources that Euclid classifies as neither point-like nor spurious. | Near the detection limit, a faint compact galaxy is classified as point-like and lost; no ground truth allows correcting this. |
| 2 | Run `copaings` on the Rubin catalogue against the Euclid one (friends-of-friends grouping with a linking length of **3 arcsec**, then ellipse overlap test) and keep only objects with blending entropy `S_b = 0`, i.e. matched one-to-one after this test. | This single condition replaces the cross-match between the two surveys, the isolation criterion and the rejection of shredded galaxy pieces. It removes around 30 % of galaxies, which can be repaired by reweighting in magnitude × log size **provided the pre-cut catalogue is kept**. But "isolated" means isolated in the sense of both surveys: a blend that neither resolves survives, and that is the floor of 0.01 to 0.02 in ellipticity. And a clumpy galaxy that one of the surveys has shredded into pieces is wrongly rejected, a morphology bias that reweighting does not repair. |
| 3 | Cut the stamp in the area that the catalogue assigns to the galaxy, and require a 29-pixel margin on all four sides, as an integer comparison. | The outer edge of the survey is lost, a loss uncorrelated with galaxy properties; **to be quantified**. |
| 4 | Reject if a single pixel of the box is flagged as unexposed, in any of the six colours. | Chosen strictness: an unexposed pixel far from the galaxy would probably not matter; **to be quantified**. |
| 5 | Reject if a damaged pixel (saturation, cosmic ray, sensor edge) falls within **`k` times the cut 2 ellipse**, the same as in the overlap test, with `k` of order 3. | `k` must be calibrated where the ellipticity shift caused by a bad pixel falls below 0.02; the list of flags is a choice; **to be quantified**. |
| 6 | Keep the galaxy if `i_cModelFlux / i_cModelFluxErr` exceeds 5, and record this value. | Untested floor; and shape measurement never selects, so shape performance will only be quantified on measurable galaxies, hence the need for a second, pixel-based metric; **to be quantified**. |
| 7 | Reject if the six astrometric solutions do not send the galaxy position to the same pixel. | Should never trigger: if it happens often, the files are mismatched; **to be quantified** as a diagnostic. |
| 8 | Cut a diagnostic stamp twice as large, measure the galaxy in **circular** apertures of 29 and 58 pixels radius, reject if the ellipticity differs by more than 0.02. | Never applied: it is the only cut whose order of magnitude of effect is unknown, **to be evaluated first**. |

**Why 3 arcsec and not 2.** It is the value recommended by `copaings`, measured rather than chosen: at 2 arcsec,
about 10 % of the blends that Euclid identifies slip through. Note, however, that the linking length only gathers
candidates, and it is the ellipse overlap that decides. The effective isolation radius is therefore that of the
ellipses, much smaller than 3 arcsec.

**A single ellipse for the whole pipeline.** `copaings` defines the ellipse from the second moments of the
*i*-band flux, with the $2\ln 2$ factor that places it at half light, and the moments of detections already include
the PSF, so the ellipse of a tiny galaxy never shrinks below the PSF size. This same ellipse is used for cut 5 and
for masking, instead of a radius in arcseconds: its area grows with apparent size, which introduces a size
dependence that reweighting can repair; and at fixed effective radius its area does not depend on ellipticity, so
it does **not** introduce the one bias that cannot be repaired.

**Two technical steps not to get wrong.** The astrometric solution returns (column, row) whereas an array is
indexed [row, column]: swapping them transposes the stamp and exchanges the two ellipticity components. And the
origin conventions differ by one pixel between FITS and arrays, which shifts the centring.

## Companion masking, and the decision that goes with it

Every companion present in the box without belonging to the target's group is masked over the extent of its
ellipse, and the masked area is filled with a piece of empty sky cut from the same area and the same colour, never
white noise. The image then shows a lone galaxy, which is what it is supposed to be. Same treatment in the cut 8
diagnostic stamp, so that the only difference between the two apertures is their size.

**And a decision, which also makes it a cut.** The target ellipticity is measured before and after masking, and the
offset is recorded. **If the offset exceeds 0.02, the stamp is discarded**; otherwise it is kept, masked. Masking
does not redeem contamination: if a companion shifts the shape, its light enters the area where the target is
measured, and masking that area erases the target along with it.

Cut 2 does not exempt from this test. Its effective radius is that of the ellipse overlap, a few tenths of an
arcsecond, whereas the weighting of the shape measurement extends two to three times further: there remains a band
where a companion does not touch the target but shifts its shape. This test catches that band.

**And the fraction of masked pixels is recorded per stamp**, to check afterwards that the network's performance does
not depend on it.

## To record without cutting

| quantity | why |
|---|---|
| blending entropy `S_b` of each object | keeping the value, not only the condition `S_b = 0`, allows tightening or loosening isolation without rebuilding the dataset |
| Euclid redshift quality flag | a galaxy with a good image and a doubtful redshift remains a good example for shape; the flag is read task by task at training time |
| PSF size and two ellipticity components, per colour | the Rubin PSF is not round, the simulations' one is, and the corresponding leakage has already been measured; without these columns it can be neither demonstrated nor corrected |
| galaxy/PSF resolution factor | tells when the measured ellipticity is dominated by the instrument; used as a weight, never as a criterion |
| signal-to-noise ratio, ellipticity offset at masking, offset between the two cut 8 apertures, ellipse semi-axes and angle, local coadd depth | all thresholds thus remain revisable without rebuilding the dataset |
