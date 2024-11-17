import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple

from .utils import *

def decay_curve( t_vals: NDArray,
                 a_vals: NDArray,
                 fs: float,
                 ir_len: int,
                 add_noise: bool = False) -> Tuple[NDArray, NDArray]:
    # check that a_vals and t_vlas have the same shape
    assert a_vals.shape == t_vals.shape, 'a_vals and t_vals must have the same shape'

    time_axis = np.linspace(0, (ir_len - 1) / fs, ir_len)
    envelope_kernel = decay_kernel(t_vals,
                                time_axis,
                                fs,
                                normalize_envelope=False,
                                add_noise=add_noise)
    
    return a_vals*envelope_kernel

def shaped_wgn( t_vals: NDArray,
                a_vals: NDArray,
                fs: float,
                ir_len: int,
                f_bands: Optional[ArrayLike] = None) -> Tuple[NDArray, NDArray]:
    """
    Synthesise RIRs with white noise shaping

    Args:
        t_vals (NDArray): desired T60 values in seconds of size n_rir x n_slopes x n_bands
        a_vals (NDArray): desired amplitudes for each slope of size n_rir x n_slopes x n_bands
        ir_len (int): Length of the IR in samples
        f_bands (Optional, ArrayLike): frequency bands in which T60s and amplitudes are specified
        n_modes (Optional, int): number of modes to synthesise if using modal synthesis
    Returns:
        NDArray, NDArray: array of RIRs of of size n_rir x ir_len x n_slopes x n_bands, and summed RIRs of size n_rir x ir_len
    """
    # assert input dimensions
    assert len(t_vals.shape) >= 1 and len(
        t_vals.shape
    ) <= 3, 'Incorrect dimension for t_vals. Must be either [n_rir x n_slopes x n_bands] or [n_rir x n_slopes] or [n_rir].'
    assert len(a_vals.shape) == len(
        t_vals.shape
    ) <= 3, 'Incorrect dimension for a_vals. Must be the same as t_vals.'

    # expand dimensions if necessary
    t_vals, a_vals, n_bands = slope_param_shape_check(t_vals, a_vals, f_bands)

    n_rirs, n_slopes = t_vals.shape[:2]
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)  # time arrray

    # initialize output arrays
    gaussian_noise = np.zeros(
        (n_rirs, ir_len, n_slopes, n_bands)) 
    shaped_noise = np.zeros_like(gaussian_noise)
    rirs = np.zeros_like(gaussian_noise)

    # envelope is in linear scale, not quadratic, therefore decay rates halve, 
    # and T values double
    t_vals_envelope = 2 * np.array(t_vals)
    a_vals_envelope = np.sqrt(a_vals)

    for i_slope in range(n_slopes):

        # generate decay envelope
        envelopes = decay_kernel(t_vals_envelope[:,i_slope,...],
                                 time,
                                 fs,
                                 normalize_envelope=True,
                                 add_noise=False)

        # generate random sequence of Gaussian noise
        random_sequence = np.random.randn(n_rirs, ir_len, 1)

        if n_bands > 1:
            # fitler the random sequence in frequency to extract the band
            gaussian_noise[:, :, i_slope, :] = octave_filtering(
                random_sequence[..., 0], 
                fs, 
                f_bands,
                ir_len=ir_len,
                compensate_filter_energy=True)

            for i_band in range(n_bands):
                # filtered gaussian noise, weighted by envelope in current band
                shaped_noise[..., i_slope, i_band] = np.einsum(
                    'nt, nt -> nt', gaussian_noise[..., i_slope, i_band],
                    envelopes[..., i_band])
                rirs[:, :, i_slope, i_band] = shaped_noise[..., i_slope, i_band] * a_vals_envelope[..., i_slope, i_band]
        else:

            # shape the random sequence and apply the envelope
            rirs[..., i_slope, :] = np.einsum(
                'ntb, nb -> ntb', random_sequence * envelopes,
                a_vals_envelope[:, i_slope, :])

    return rirs, rirs.sum(axis=-1).sum(axis=-1)


