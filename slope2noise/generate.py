import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple
from loguru import logger
import pyfar as pf
from .utils import *


def decay_curve(
    t_vals: NDArray,
    a_vals: NDArray,
    n_vals: NDArray,
    fs: float,
    ir_len: int,
    add_noise: bool = False,
) -> Tuple[NDArray, NDArray]:
    # check that a_vals and t_vlas have the same shape
    assert a_vals.shape == t_vals.shape, "a_vals and t_vals must have the same shape"

    time_axis = np.linspace(0, (ir_len - 1) / fs, ir_len)
    envelope_kernel = decay_kernel(t_vals,
                                   time_axis,
                                   fs,
                                   normalize_envelope=False,
                                   add_noise=add_noise)
    if add_noise:
        # concatenate noise term to a_vals
        a_vals = np.concatenate(
            (a_vals, np.expand_dims(n_vals / ir_len, axis=-1)), axis=-1)

    return a_vals * envelope_kernel


def shaped_wgn(
    t_vals: NDArray,
    a_vals: NDArray,
    fs: float,
    ir_len: int,
    num_fractions: int = 1,
    n_vals: Optional[NDArray] = None,
    f_bands: Optional[ArrayLike] = None,
    use_amp_preserving_filterbank: Optional[bool] = True,
    verbose: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Synthesise RIRs with white noise shaping

    Args:
        t_vals (NDArray): desired T60 values in seconds of size n_rir x n_slopes x n_bands
        a_vals (NDArray): desired amplitudes for each slope of size n_rir x n_slopes x n_bands
        fs (float): sampling frequency
        ir_len (int): Length of the IR in samples
        num_fractions (int): fractions in fractional octave filterbank for filtering white noise
        n_vals (Optional, NDArray): desired noise floor for each band of size n_rir x n_bands
        f_bands (Optional, ArrayLike): frequency bands in which T60s and amplitudes are specified
        use_amp_preserving_filterbank (Optional, bool): whether to use Pyfar's perfect reconstruction 
                                                        octave filterbank, or energy preserving filterbank
    Returns:
        NDArray, NDArray: array of RIRs of of size n_rir x ir_len x n_slopes x n_bands, and summed RIRs of size n_rir x ir_len
    """
    # assert input dimensions
    assert (
        len(t_vals.shape) >= 1 and len(t_vals.shape) <= 3
    ), "Incorrect dimension for t_vals. Must be either [n_rir x n_slopes x n_bands] or [n_rir x n_slopes] or [n_rir]."
    assert (len(a_vals.shape) == len(t_vals.shape) <=
            3), "Incorrect dimension for a_vals. Must be the same as t_vals."

    if n_vals is not None:
        assert len(n_vals.shape) == len(
            t_vals.shape
        ) - 1 <= 2, 'Incorrect dimension for n_vals. Must be the same be either  [n_rir x n_bands] or [n_rir].'
        n_vals_envelope = -np.sqrt(n_vals / ir_len)
        if len(n_vals.shape) == 1:
            n_vals_envelope = np.expand_dims(n_vals_envelope, axis=-1)
            
    # expand dimensions if necessary
    t_vals, a_vals, n_bands = slope_param_shape_check(t_vals, a_vals, f_bands)

    n_rirs, n_slopes = t_vals.shape[:2]
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)  # time arrray

    # initialize output arrays
    gaussian_noise = np.zeros((n_rirs, ir_len, n_slopes + 1, n_bands))
    rirs = np.zeros_like(gaussian_noise)

    # envelope is in linear scale, not quadratic, therefore decay rates halve,
    # and T values double
    t_vals_envelope = 2 * np.array(t_vals)
    a_vals_envelope = np.expand_dims(np.sqrt(a_vals), 1)

    loop_range = n_slopes if n_vals is None else n_slopes + 1

    for i_slope in range(loop_range):
        # NOTE: last slope index is interpreted as noise term
        # generate decay envelope
        if i_slope < n_slopes:
            envelopes = decay_kernel(
                t_vals_envelope[:, i_slope, ...],
                time,
                fs,
                normalize_envelope=True,
                add_noise=False,
            )
        if verbose:
            logger.info(f"Done with kernel generation for slope {i_slope+1}")
        # generate random sequence of Gaussian noise, and filter it
        random_sequence = np.random.randn(n_rirs, ir_len, 1)

        if n_bands > 1:
            if verbose:
                logger.info(f"Filtering noise into subbands slope {i_slope+1}")
            # this is of shape n_rirs x ir_len x n_bands
            filtered_gaussian_noise = octave_filtering(
                random_sequence[..., 0],
                fs,
                f_bands,
                num_fractions=num_fractions,
                ir_len=ir_len,
                compensate_filter_energy=True,
                use_amp_preserving_filterbank=use_amp_preserving_filterbank,
            )
            if verbose:
                logger.info(f"Done with octave filtering")

            if i_slope < n_slopes:
                rirs[:, :, i_slope, :] = np.einsum(
                    'ntb, ntb -> ntb', filtered_gaussian_noise * envelopes,
                    a_vals_envelope[..., i_slope, :])
            else:
                rirs[:, :,
                     i_slope, :] = np.einsum(
                         'ntb, nb -> ntb', filtered_gaussian_noise,
                         n_vals_envelope)
        else:
            # shape the random sequence and apply the envelope
            if i_slope < n_slopes:
                rirs[...,
                     i_slope, :] = np.einsum('ntb, nb -> ntb',
                                             random_sequence * envelopes,
                                             a_vals_envelope[:, 0, i_slope, :])
            else:
                rirs[...,
                     i_slope, :] = np.einsum('ntb, nb -> ntb', random_sequence,
                                             n_vals_envelope)
        if verbose:
            logger.info(f"Done with noise shaping for slope {i_slope+1}")

    return rirs, rirs.sum(axis=-1).sum(axis=-1)
