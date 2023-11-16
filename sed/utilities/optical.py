""" Functions related to the optical laser excitation of the sample."""
from typing import Sequence
from typing import Union

import numpy as np


def absorbed_energy_density(
    fluence: float,
    reflectivity: float,
    penetration_depth: float = None,
) -> float:
    """Absorbed energy density from fluence

    This function calculates the absorbed energy density from the fluence, the reflectivity and
    the penetration depth. If the penetration depth is not provided, the absorbed energy density
    is returned in surface units (mJ/cm²). If the penetration depth is provided, the absorbed
    energy density is returned in volume units (J/cm³).

    Args:
        fluence: fluence in mJ/cm²
        reflectivity: sample reflectivity at the frequency of the pump beam
        penetration_depth: optical penetration depth at the frequency of the pump beam in nanometers
            if none, the absorbed_energy_density in surface units is returned

    Returns:
        absorbed energy density in  mJ/cm² or J/cm³ if the penetration depth is provided
    """
    aed: float = fluence * (1 - reflectivity)
    if penetration_depth is not None:
        pdepth_cm = penetration_depth * 1e-7
        aed /= 1000 * pdepth_cm
    return aed


def effective_gaussian_area(
    sigma: Union[float, Sequence[float]],
    photoemission_order: int = 1,
    sigma_is_fwhm=False,
) -> float:
    """Effective area of a 2D gaussian.

    This function calculates the effective area of a 2D gaussian beam, taking into account the
    photoemission order. The effective area is defined as the area of a circle with the same
    intensity as the gaussian beam.
    TODO: cite correctly M. Dendzik et al. Supplementary Information

    Args:
        sigma Union[float, Sequence[float]]: gaussian sigma in µm. If a sequence is passed, the
            first value is interpreted as sigma_x and the second as sigma_y. If only one value is
            passed, sigma_y is assumed to be equal to sigma_x. If sigma_is_fwhm is true, sigma and
            sigma_y are interpreted as FWHM
        photoemission_order (int): order of the photoemission process
        sigma_is_fwhm (bool): if True, sigma and sigma_y are interpreted as FWHM. Default: False

    Returns:
        area (float): effective illumination area in µm²
    """
    if isinstance(sigma, Sequence):
        sigma_x: float = sigma[0]
        sigma_y: float = sigma[1]
    elif isinstance(sigma, (float, np.floating, int, np.integer)):
        sigma_x = sigma
        sigma_y = sigma
    else:
        raise TypeError(f"sigma must be a float or a sequence of floats, not {type(sigma)}")
    if sigma_is_fwhm:
        FWHM_x: float = sigma_x
        FWHM_y: float = sigma_y
    else:
        FWHM_x = sigma_to_fwhm(sigma_x)
        FWHM_y = sigma_to_fwhm(sigma_y)
    return (photoemission_order * FWHM_x * FWHM_y * np.pi) / (4 * np.log(2))


def fluence(
    pulse_energy: float,
    area: float,
) -> float:
    """Fluence from pulse energy and area.

    This function calculates the fluence from the pulse energy and the illuminated area.

    Args:
        pulse_energy: photon pulse energy in µJ
        area: illuminated area in µm²

    Returns:
        fluence in mJ/cm²
    """
    return pulse_energy * 1e5 / area


def sigma_to_fwhm(sigma: float) -> float:
    """Get fwhm from the standard deviation of the gaussian distribution

    Args:
        sigma: standard deviation of the gaussian distribution

    Returns:
        fwhm: full width at half maximum of the gaussian distribution
    """
    return sigma * 2 * np.sqrt(2 * np.log(2))


def fwhm_to_sigma(fwhm: float) -> float:
    """Get the standard deviation of the gaussian distribution from its FWHM

    Args:
        fwhm: full width at half maximum of the gaussian distribution

    Returns:
        sigma: standard deviation of the gaussian distribution
    """
    return fwhm / 2 * np.sqrt(2 * np.log(2))
