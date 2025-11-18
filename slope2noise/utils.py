import numpy as np
import pyfar as pf
from numpy.typing import NDArray, ArrayLike
from typing import Union, List, Optional
from scipy.signal import butter, zpk2sos, sosfreqz, sosfilt, fftconvolve
from scipy.optimize import lsq_linear, least_squares
from tqdm import tqdm
import soundfile as sf
from loguru import logger


def save_audio(filepath: str, x: NDArray, fs: int = 48000):
    """Save audio to file using soundfile"""
    sf.write(filepath, x, fs)


def discard_trailing_zeros(rir: NDArray, axis: int = -1) -> NDArray:
    """Discard trailing zeros from the RIR."""
    # find first non-zero element from back
    last_above_thres = rir.shape[axis] - np.argmax(
        (np.flip(rir, axis=axis) != 0)).squeeze().astype(int)
    # discard from that sample onwards
    out = rir[..., :last_above_thres]
    return out


def discard_last_n_percent(edc: NDArray, n_percent: float) -> NDArray:
    """Discard last n% from the EDC."""
    last_id = (np.round((1 - n_percent / 100) * edc.shape[-1])).astype(int)
    out = edc[..., 0:last_id]
    return out


def slope_param_shape_check(t_vals: NDArray,
                            a_vals: NDArray,
                            n_vals: Optional[ArrayLike] = None,
                            f_bands: Optional[ArrayLike] = None) -> tuple[NDArray, NDArray, int]:
    """
    Check and adjust the shape of t_vals and a_vals for slope parameter calculation.

    Parameters:
    t_vals (NDArray): Decay time values array.
    a_vals (NDArray): Amplitude values array.
    n_vals (Optional, ArrayLike): Noise values array, if applicable.    
    f_bands (Optional, ArrayLike): Frequency bands array, if applicable.

    Returns:
    tuple: Adjusted t_vals, a_vals, and the number of bands.
    """

    # basic asserts for t_vals and a_vals shapes (allow 1..3 dims)
    assert (
        len(t_vals.shape) >= 1 and len(t_vals.shape) <= 3
    ), "Incorrect dimension for t_vals. Must be either [n_rir x n_slopes x n_bands] or [n_rir x n_slopes] or [n_rir]."
    assert (len(a_vals.shape) == len(t_vals.shape) <= 3), "Incorrect dimension for a_vals. Must be the same as t_vals."

    # handle optional n_vals: check dims and expand last axis if needed
    if n_vals is not None:
        assert len(n_vals.shape) == len(t_vals.shape) - 1 <= 2, (
            "Incorrect dimension for n_vals. Must be either [n_rir x n_bands] or [n_rir]."
        )
        if len(n_vals.shape) == 1:
            n_vals = np.expand_dims(n_vals, axis=-1)

    # expand t_vals/a_vals to 3 dims (n_rir x n_slopes x n_bands) when needed
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
        ), "Mismatch in number of bands. t_vals should have appropriate dimensions."
        n_bands = len(f_bands)
    else:
        n_bands = 1  # broadband

    return t_vals, a_vals, n_vals, n_bands


def db(x: ArrayLike,
       is_squared: bool = False,
       min_value: float = -200) -> ArrayLike:
    """Convert values to decibels.

    Args:
        x (ArrayLike):
            value(s) to be converted to dB.
        is_squared (bool):
            Indicates whether `x` represents some power-like quantity (True) or some root-power-like quantity (False).
            Defaults to False, i.e. `x` is a root-power-like auqntity (e.g. Voltage, pressure, ...).
        min_value (float): cap the decibels to this value, cannot be lower

    Returns:
        An array with the converted values, in dB.
    """
    factor = 10.0 if is_squared else 20.0

    x = np.abs(x)
    y = factor * np.log10(x + np.finfo(np.float32).eps)

    return y.clip(min=min_value)


def ms_to_samps(ms: Union[float, ArrayLike],
                fs: float) -> Union[int, ArrayLike]:
    """
    Convert ms to samples
    Args:
        ms (float or ArrayLike): time in ms
        fs (float): sampling rate
    Returns:
        int, ArrayLike: time in samples
    """
    samp = ms * 1e-3 * fs
    if np.isscalar(samp):
        return int(samp)
    else:
        return samp.astype(np.int32)


def schroeder_backward_int(rir: NDArray,
                           time_axis: int = -1,
                           normalize: bool = False,
                           discard_last_zeros: bool = False):

    if discard_last_zeros:
        out = discard_trailing_zeros(rir)
    else:
        out = rir

    # Backwards integral
    out = np.flip(out, axis=time_axis)
    out = np.cumsum(out**2, axis=time_axis)
    out = np.flip(out, axis=time_axis)

    if normalize:
        # Normalize to 1
        norm_vals = np.max(out, axis=time_axis, keepdims=True)  # per channel
        # norm_vals = np.max(out, keepdims=True)  # global max for all channels
        out = out / norm_vals if not np.isnan(norm_vals).any() else out
        return out
    else:
        return out


def decay_kernel(
    envelope_t: Union[float, ArrayLike],
    time: ArrayLike,
    fs: float,
    normalize_envelope: bool = False,
    add_noise: bool = False,
) -> NDArray:
    """
    Decay kernel for the exponential envelope. Accepts only one frequncy band at a time.
    Args:
        envelope_t: the T60 values (doubled)
        time (ArrayLike): time vector
        fs (float): sampling rate
        normalize_envelope (bool): whether to normalise the energy to account for shroeder integration
        add_noise (bool): whether to add noise to the decay kernel (use it only if you're modelling only one slope)
    Returns:
        NDArray: exp(-t/tau) exponential decay kernel, or exp(-t/tau) + n(t) with noise
    """
    assert len(envelope_t.shape) <= 2, "Only one frequency band is accepted"

    tau_vals = np.log(10**6) / envelope_t
    exponential = np.exp(-np.einsum("nb,t->ntb", tau_vals, time))

    # normalise the kernel to have unit energy
    if normalize_envelope:
        exponential = np.einsum("ntb, nb -> ntb", exponential,
                                np.sqrt((1 - np.exp(-2 * tau_vals / fs))))

    # construct the decay kernel
    if add_noise:
        # calculate noise
        ir_len = len(time)
        noise = np.linspace(ir_len, 0, ir_len)
        noise = np.expand_dims(noise, axis=(0, -1))
        noise = np.tile(
            noise, (exponential.shape[0], 1, 1))  # repeat noise along all rirs
        return np.concatenate((exponential, noise), axis=-1)
    else:
        return exponential


def calculate_energy_envelope(sig: NDArray,
                              fs: float,
                              smooth_time_ms: float,
                              time_axis: int = -1) -> ArrayLike:
    """
    Calculate the energy envelope (broadband EDC) of a RIR
    Args:
        sig (ArrayLike): ND RIR signal
        fs (float): sampling rate
        smooth_time_ms (float): smoothing window length in ms,
                                longer window leads to more smoothing
        time_axis (int): axis along which time samples are specified
    """
    staps = ms_to_samps(smooth_time_ms / 2, fs)
    odd_win_len = 2 * staps - 1
    # normalised smoothing window
    bs = np.hanning(odd_win_len) / np.sum(np.hanning(odd_win_len))

    # Reshape bs to broadcast correctly along the time axis
    bs_shape = [1] * sig.ndim
    bs_shape[time_axis] = bs.shape[0]
    bs_broadcasted = bs.reshape(bs_shape)

    # Pad along the time axis, before and after the signal
    pad_width = [(0, 0)] * sig.ndim
    pad_width[time_axis] = (staps, staps)
    padded_signal = np.pad(np.power(sig, 2), pad_width, mode='constant')

    # Smooth using convolution along the time axis
    smoothed_signal = fftconvolve(padded_signal,
                                  bs_broadcasted,
                                  mode='valid',
                                  axes=time_axis)

    # Take square root to get envelope
    env = np.sqrt(smoothed_signal)
    env = env[..., odd_win_len:]

    return env


def calculate_amplitudes_least_squares(t_vals: NDArray,
                                       fs: float,
                                       rirs: NDArray,
                                       f_bands: Optional[ArrayLike] = None,
                                       leave_out_ms: float = 50.0,
                                       verbose: bool = False,
                                       use_non_linear_ls: bool = True,
                                       downsample_factor: int = 1) -> NDArray:
    """
    Calculate amplitudes (one for each slope) using linear least squares
    Args:
        t_vals (NDArray): the T60s of shape n_rir x n_slopes x n_bands
        fs (float): sampling rate
        rirs (NDArray): RIR matrix of shape n_rir x ir_len x n_bands
        f_bands (ArrayLike): frequency bands where RIR is calculated
        leave_out_ms (float): number of samples to leave out of the
                             RIR to prevent bad conditioning
        verbose (bool): if true, the error in subbands is displayed
        downsample_factor (int): downsample the signal after calculating EDC, otherwise
                                 calculation is too heavy
    Returns:
        NDArray: estimated amplitudes of shape n_rir x n_slopes + 1 x n_bands.
                The first slope contains the noise floor.
    """

    if rirs.ndim == 2:
        rirs = np.reshape(rirs, (1, rirs.shape[0], rirs.shape[1]))
        t_vals = t_vals.reshape(1, t_vals.shape[0], t_vals.shape[1])

    leave_out_samps = ms_to_samps(leave_out_ms, fs)
    rirs = rirs[:, :-leave_out_samps, :]

    num_rirs, ir_len, n_bands = rirs.shape
    ir_len_downsampled = ir_len // downsample_factor
    n_slopes = t_vals.shape[1]
    time = np.linspace(0, (ir_len_downsampled - 1) / fs, ir_len_downsampled)

    # find the exponential decay envelope for each slope
    # the decay kernel sum_{k=1}^K exp(-t/tau_k) of size n_rir x ir_len x n_slopes x n_bands
    envelopes = np.zeros((num_rirs, ir_len_downsampled, n_slopes + 1, n_bands))

    # the first slope contains the noise term, and has kernel L - t
    psi_0 = 0.5 * (ir_len_downsampled - np.arange(ir_len_downsampled)) / fs
    envelopes[:, :, 0, :] = np.tile(psi_0[np.newaxis, :, np.newaxis],
                                    (num_rirs, 1, n_bands))

    for i_slope in range(n_slopes):
        # envelope is in linear scale, not quadratic, therefore decay rates halve, and T values double
        envelope_t = np.array(t_vals[:, i_slope, ...])
        if num_rirs == 1:
            envelope_t = envelope_t.reshape(1, n_bands)
        # generate decay envelopes from t_vals
        envelopes[:, :,
                  i_slope + 1, :] = decay_kernel(envelope_t,
                                                 time,
                                                 fs,
                                                 normalize_envelope=False,
                                                 add_noise=False)

    est_level = np.zeros((num_rirs, n_slopes + 1, n_bands), dtype=float)
    error = np.zeros_like(est_level)

    for i in tqdm(range(num_rirs)):
        for k in range(n_bands):

            cur_rir = rirs[i, :, k]

            # calculate EDC of the RIR
            cur_edc = schroeder_backward_int(cur_rir,
                                             normalize=False,
                                             discard_last_zeros=False)
            cur_edc = cur_edc[::downsample_factor].reshape(
                ir_len_downsampled, 1)

            # get the current RIR's envelope
            cur_envelope = np.zeros((ir_len_downsampled, n_slopes + 1))
            cur_envelope[:, 0] = envelopes[i, :, 0, k]
            # psi_k(t) - psi_k(L)
            cur_envelope[:, 1:] = (envelopes[i, :, 1:, k] -
                                   envelopes[i, -1, 1:, k])
            assert cur_envelope.shape == (ir_len_downsampled, n_slopes + 1)

            if use_non_linear_ls:
                # non-linear least squares minimising error in dB,
                # same as Georg's implementation
                residuals = lambda params: (
                    db(cur_envelope @ params[..., np.newaxis], is_squared=True
                       ) - db(cur_edc, is_squared=True)).squeeze()
                params_init = np.r_[1e-10, np.ones(n_slopes)]
                result = least_squares(residuals,
                                       params_init,
                                       bounds=(np.zeros(n_slopes + 1),
                                               np.r_[1,
                                                     10 * np.ones(n_slopes)]))
                cur_level = result.x.reshape(n_slopes + 1, 1)

            else:
                # linear least squares without constraints
                cur_level = np.linalg.pinv(cur_envelope) @ cur_edc

            if verbose:
                error[i, :, k] = np.linalg.norm(cur_envelope @ cur_level -
                                                cur_edc)**2
                logger.info(
                    f"num_rir = {i}, num_band = {k}, error = {db(error[i,:,k], is_squared=True)} dB"
                )
            est_level[i, :, k] = np.squeeze(cur_level)

    est_amps = est_level
    return est_amps


def octave_filtering(
        input_signal: Union[ArrayLike, NDArray],
        fs: float,
        f_bands: List,
        get_filter_ir: bool = False,
        compensate_filter_energy: bool = False,
        num_fractions: int = 1,
        filter_length: int = 4096,
        ir_len: Optional[int] = None,
        use_amp_preserving_filterbank: Optional[bool] = False) -> NDArray:
    """
    Apply an octave bandpass filter to the input signal.

    Parameters:
    input_signal (np.ndarray): The input signal to be filtered, of shape n_rir x ir_len, or ir_len
    fs (float): The sampling frequency of the input signal.
    f_bands (List): List of frequency bands for filtering.
    ir_len (Optional[int]): Length of the impulse response of the filters.
    get_filter_ir (bool, optional): Whether to get the filter impulse response. Default is False.
    use_amp_preserving_filterbank (bool, optional): if using pyfar, whether to use amp preserving FIR filterbank or
                                                    energy preserving Butterworth filterbank

    Returns:
    np.ndarray: The filtered signal.
    """
    num_bands = len(f_bands)
    if ir_len is None:
        ir_len = len(
            input_signal) if input_signal.ndim == 1 else input_signal.shape[-1]

    assert (num_fractions == 1) | (num_fractions
                                   == 3), "num_fractions must be either 1 or 3"

    out_bands = np.zeros((*input_signal.shape, num_bands))
    if get_filter_ir:
        impulse_response = np.zeros((num_bands, *input_signal.shape))

    pf_freqs, _ = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=num_fractions, frequency_range=(f_bands[0], f_bands[-1]))
    assert np.allclose(np.array(f_bands),
                       pf_freqs), "centre frequencies don't match"

    if use_amp_preserving_filterbank:
        subband_filters, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None,
            num_fractions=num_fractions,
            frequency_range=(f_bands[0], f_bands[-1]),
            sampling_rate=fs,
            n_samples=int(filter_length),
        )
        impulse_response = subband_filters.coefficients
    else:
        subband_filters = pf.dsp.filter.fractional_octave_bands(
            None,
            num_fractions=num_fractions,
            frequency_range=(f_bands[0], f_bands[-1]),
            sampling_rate=fs,
        )
        impulse_response = np.zeros((num_bands, ir_len))

    for b_idx in range(num_bands):
        # FIR filterbank
        if use_amp_preserving_filterbank:
            if input_signal.ndim > 1:
                cur_filters = np.tile(subband_filters.coefficients[b_idx, ...],
                                      (input_signal.shape[0], 1))
            else:
                cur_filters = subband_filters.coefficients[b_idx, ...]

            out_bands[..., b_idx] = fftconvolve(input_signal,
                                                cur_filters,
                                                axes=-1,
                                                mode='same')
            if compensate_filter_energy:
                out_bands[..., b_idx] /= np.sqrt(
                    np.sum(impulse_response[b_idx, ...]**2))
        else:
            impulse_response[b_idx, :] = sosfilt(
                subband_filters.coefficients[b_idx, ...],
                np.r_[1.0, np.zeros(ir_len - 1)])
            if get_filter_ir:
                out_bands[..., b_idx] = impulse_response
            else:
                cur_filters = subband_filters.coefficients[b_idx, ...]
                out_bands[..., b_idx] = sosfilt(cur_filters,
                                                input_signal,
                                                axis=-1)

                if compensate_filter_energy:
                    out_bands[..., b_idx] /= np.sqrt(
                        np.sum(impulse_response[b_idx, :]**2))

    if get_filter_ir:
        return out_bands, impulse_response
    else:
        return out_bands
