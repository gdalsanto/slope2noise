import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union, Optional, Tuple
from loguru import logger

from .utils import *


def rir_synthesis(t_vals: NDArray,
                  a_vals: NDArray,
                  f_bands: ArrayLike,
                  fs: float,
                  ir_len: int,
                  type: str = 'modal',
                  n_modes: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """
    Synthesise RIRs with modal synthesis or white noise shaping
    Args:
        t_vals (NDArray): desired T60 values in seconds of size n_rir x n_slopes x n_bands
        a_vals (NDArray): desired amplitudes for each slope of size n_rir x n_slopes x n_bands
        f_bands (ArrayLike): frequency bands in which T60s and amplitudes are specified
        ir_len (int): Length of the IR in samples
        type (str): method used for synthesis, modal or additive white noise
        n_modes (optional, int): number of modes to synthesise if using modal synthesis
    Returns:
        NDArray, NDArray: array of RIRs of of size n_rir x ir_len x n_slopes x n_bands, and summed RIRs of size n_rir x ir_len
    """
    # frequency bands are assumend to be log spaced

    # assert input dimensions
    assert len(t_vals.shape) >= 1 and len(
        t_vals.shape
    ) <= 3, 'Incorrect dimension for t_vals. Must be either [n_rir x n_slopes x n_bands] or [n_rir x n_slopes] or [n_rir].'
    assert len(a_vals.shape) == len(
        t_vals.shape
    ) <= 3, 'Incorrect dimension for a_vals. Must be the same as t_vals.'

    # expand dimensions if necessary
    if len(t_vals.shape) < 3:
        t_vals = np.reshape(
            t_vals,
            (*t_vals.shape, *tuple([1 for d in range(3 - len(t_vals.shape))])))
        a_vals = np.reshape(
            a_vals,
            (*a_vals.shape, *tuple([1 for d in range(3 - len(a_vals.shape))])))
    if f_bands is not None:
        assert t_vals.shape[-1] == len(
            f_bands
        ), 'Mismatch in number of bands. t_vals should have appropriate dimensions.'
        if type == 'modal':
            # modal synthesis requires side bands for interpolation
            n_bands = len(f_bands) + 2
        else:
            n_bands = len(f_bands)
    else:
        n_bands = 1  # broadband

    n_rirs = t_vals.shape[0]  # number of rir to generate
    n_slopes = t_vals.shape[1]  # number of common slopes
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)  # time arrray

    # initialize output arrays
    gaussian_noise = np.zeros(
        (n_rirs, ir_len, n_slopes, n_bands))  # without envelopes
    envelope_a = np.zeros_like(gaussian_noise)
    synthesis_rirs = np.zeros_like(gaussian_noise)
    filtered_noise = np.zeros_like(gaussian_noise)

    # envelope is in linear scale, not quadratic, therefore decay rates halve, and T values double
    t_vals_envelope = 2 * np.array(t_vals)
    a_vals_envelope = np.sqrt(a_vals)

    for i_slope in range(n_slopes):
        # generate decay envelope
        envelopes = decay_kernel(t_vals_envelope[:, i_slope, ...],
                                 time,
                                 fs,
                                 normalise_envelope=True,
                                 add_noise=False)

        logger.info(f"Done with kernel generation for slope {i_slope+1}")

        if n_bands > 1 and type == 'modal':
            # duplicate first and last envelope for residual band
            envelopes = np.concatenate([
                np.expand_dims(envelopes[:, :, 0], -1), envelopes,
                np.expand_dims(envelopes[:, :, -1], -1)
            ],
                                       axis=-1)
            a_vals_envelope = np.concatenate([
                np.expand_dims(a_vals_envelope[:, :, 0], 1), a_vals_envelope,
                np.expand_dims(a_vals_envelope[:, :, -1], 1)
            ],
                                             axis=-1)

        if type == 'modal':

            # convert envelope to dB
            envelopes_db = 10 * np.log10(envelopes)

            # random mode phase
            mode_phase = 2 * np.pi * np.random.rand(n_rirs, n_modes)
            # randomly draw log-spaced frequencies
            mode_freq = np.expand_dims(np.logspace(np.log10(1),
                                                   np.log10(0.99 * fs / 2),
                                                   n_modes),
                                       axis=0)

            # # add jitter TODO fix jitter (as of now cannot be done this way for n_rirs > 1)
            # jitter = np.mean(np.diff(np.log10(mode_freq[0,:]), axis=-1))
            # mode_freq = 10 ** (np.log10(mode_freq) + np.random.rand(n_rirs, n_modes) * jitter)
            # remove mode frequencies close to aliasing
            # mode_freq = mode_freq[mode_freq < 0.99 * fs / 2]
            # n_modes = len(mode_freq)

            # for mode frequencies between band centers: interpolate position in log
            # scale and combine their bands' contributions. Decay rate of
            # "in-between" modes should be in between the center frequency rates
            band_fraction = np.interp(np.log10(mode_freq),
                                      np.log10([1, *f_bands, fs / 2]),
                                      np.arange(1, n_bands + 1)).squeeze()
            i_band = np.round(band_fraction).astype(int) - 1
            band_floor = np.floor(band_fraction).astype(int)
            band_mix = band_fraction - band_floor

            for i_mode in range(n_modes):
                # Sinusoidal mode with specified frequency and phase
                # TODO add aption either mode or wgn
                mode = np.sin(
                    2 * np.pi *
                    np.einsum('n, t -> nt', mode_freq[:, i_mode], time) +
                    np.expand_dims(mode_phase[:, i_mode], axis=-1))

                gaussian_noise[:, :, i_slope, i_band[i_mode]] += mode

                # Interpolate envelope in dB scale
                inter_envelope_db = (
                    1 - band_mix[i_mode]
                ) * envelopes_db[:, :, band_floor[i_mode] - 1] + band_mix[
                    i_mode] * envelopes_db[:, :, band_floor[i_mode]]
                interp_envelope = 10**(inter_envelope_db / 10
                                       )  # Convert back to linear scale

                # Apply envelope to the sinusoidal mode
                mode = mode * interp_envelope

                # Save amplitude envelopes
                envelope_a[:, :, i_slope,
                           i_band[i_mode]] = np.expand_dims(
                               np.sqrt(a_vals[:, i_slope, i_band[i_mode]]),
                               axis=-1) * np.sqrt((1 - interp_envelope))

                # Add the mode to the corresponding band, and weight by the amplitude envelope
                synthesis_rirs[:, :, i_slope, i_band[
                    i_mode]] += mode * envelope_a[:, :, i_slope,
                                                  i_band[i_mode]]

        elif type == 'wgn':

            # generate random sequence of Gaussian noise
            random_sequence = np.random.randn(n_rirs, ir_len, 1)

            if n_bands > 1:
                # get energy of the filter bank
                impulse = np.zeros((ir_len))
                impulse[0] = 1
                # the input inpulse will not be used in this case actually, get_filter argument is just a quick fix
                ir_octave_filter = octave_filtering(impulse,
                                                    fs,
                                                    f_bands,
                                                    get_filter=True)
                band_energy = sum(ir_octave_filter**2, 0)
                # fitler the random sequence in frequency to extract the band
                # this is of shape n_rirs x ir_len x n_slopes x n_bands
                logger.info(
                    f"Filtering noise into subbands for slope {i_slope+1}")
                filtered_noise[:, :, i_slope, :] = octave_filtering(
                    random_sequence[..., 0], fs, f_bands)
                logger.info(
                    f"Done with octave filtering for slope {i_slope+1}")

                for i_band in range(n_bands):
                    # filtered gaussian noise, weighted by envelope in current band
                    gaussian_noise[:, :, i_slope, i_band] = np.einsum(
                        'nt, nt -> nt', filtered_noise[:, :, i_slope, i_band],
                        envelopes[..., i_band])
                    # amplitudes weighted by the filter's energy in the band
                    envelope_a[:, :, i_slope, i_band] = np.sqrt(
                        np.expand_dims(a_vals_envelope[:, i_slope, i_band],
                                       axis=-1) / band_energy[i_band])
                    synthesis_rirs[:, :, i_slope,
                                   i_band] = gaussian_noise[:, :, i_slope,
                                                            i_band] * envelope_a[:, :,
                                                                                 i_slope,
                                                                                 i_band]

            else:

                # shape the random sequence and apply the envelope
                synthesis_rirs[:, :, i_slope, :] = np.einsum(
                    'ntb, nb -> ntb', random_sequence * envelopes,
                    np.sqrt(a_vals[:, i_slope, :]))
            logger.info(
                f"Done generating shaped white noise for slope {i_slope+1}")

    return synthesis_rirs, synthesis_rirs.sum(axis=-1).sum(axis=-1)
