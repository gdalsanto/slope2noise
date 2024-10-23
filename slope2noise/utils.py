import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Union
from scipy.signal import butter, zpk2sos, sosfreqz, sosfilt, fftconvolve
import soundfile as sf
from loguru import logger


def save_audio(filepath, x, fs=48000):
    sf.write(filepath, x, fs)


def discard_trailing_zeros(rir):
    # find first non-zero element from back
    last_above_thres = rir.shape[-1] - np.argmax(
        (np.flip(rir, axis=-1) != 0)).squeeze().astype(int)
    # discard from that sample onwards
    out = rir[..., :last_above_thres]
    return out


def discard_last_n_percent(edc, n_percent: float):
    # Discard last n%
    last_id = (np.round((1 - n_percent / 100) * edc.shape[-1])).astype(int)
    out = edc[..., 0:last_id]

    return out


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


def schroeder_backward_int(rir: NDArray, normalise: bool = True):

    out = discard_trailing_zeros(rir)

    # Backwards integral
    out = np.flip(out, axis=-1)
    out = np.cumsum(out**2, axis=-1)
    out = np.flip(out, axis=-1)

    if normalise:
        # Normalize to 1
        norm_vals = np.max(out, axis=-1, keepdims=True)  # per channel
        out = out / norm_vals

        return out, norm_vals
    else:
        return out


def decay_kernel(envelope_t: Union[float, ArrayLike],
                 time: ArrayLike,
                 fs: float,
                 normalise_envelope: bool = False,
                 add_noise: bool = True) -> NDArray:
    """
    Decay kernel for the exponential envelope
    Args:
        envelope_t: the T60 values (doubled)
        time (ArrayLike): time vector
        fs (float): sampling rate
        normalise_envelope (bool): whether to normalise the energy of the envelope to 1
        add_noise (bool): whether to add noise to the decay kernel
    Returns:
        NDArray: exp(-t/tau) exponential decay kernel, or exp(-t/tau) + n(t) with noise
    """

    tau_vals = np.log(10**6) / envelope_t
    exponential = np.exp(-np.einsum('nb,t->ntb', tau_vals, time))

    # normalise the kernel to have unit energy
    if normalise_envelope:
        exponential = np.einsum('ntb, nb -> ntb', exponential,
                                np.sqrt((1 - np.exp(-2 * tau_vals / fs))))

    # calculate noise
    ir_len = len(time)
    noise = np.linspace(1, 1 / ir_len, ir_len)

    # construct the decay kernel
    if add_noise:
        return np.concatenate((exponential, noise), axis=0)
    else:
        return exponential


def calculate_energy_envelope(sig: ArrayLike, fs: float,
                              smooth_time_ms: float) -> ArrayLike:
    """
    Calculate the energy envelope (broadband EDC) of a RIR
    Args:
        sig (ArrayLike): 1D RIR signal
        fs (float): sampling rate
        smooth_time_ms (float): smoothing window length in ms, 
                                longer window leads to more smoothing
    """
    staps = ms_to_samps(smooth_time_ms / 2, fs)
    odd_win_len = 2 * staps - 1
    # normalised smoothing window
    bs = np.hanning(odd_win_len) / np.sum(np.hanning(odd_win_len))
    # zero-pad signal on either side
    padded_signal = np.concatenate((np.zeros(staps), sig**2, np.zeros(staps)),
                                   axis=0)
    # smooth signal by convolving with window
    smoothed_signal = fftconvolve(bs, padded_signal)
    env = np.real(np.sqrt(smoothed_signal))
    # ignore the first win_len samples
    env = env[odd_win_len + np.arange(len(sig))]
    return env


def calculate_amplitudes_least_squares(t_vals: NDArray,
                                       fs: float,
                                       rirs: NDArray,
                                       leave_out_ms: float = 50.0) -> NDArray:
    """
    Calculate amplitudes (one for each slope) using linear least squares
    Args:
        t_vals (NDArray): the T60s of shape n_rir x n_slopes x n_bands
        fs (float): sampling rate
        rirs (NDArray): RIR matrix of shape n_rir x ir_len x n_bands
        leave_out_ms (float): number of samples to leave out of the 
                             RIR to prevent bad conditioning
    Returns:
        NDArray: estimated amplitudes of shape n_rir x n_slopes x n_bands
    """

    if rirs.ndim == 2:
        rirs = np.reshape(rirs, (1, rirs.shape[0], rirs.shape[1]))
        t_vals = t_vals.reshape(1, t_vals.shape[0], t_vals.shape[1])

    leave_out_samps = ms_to_samps(leave_out_ms, fs)
    rirs = rirs[:, :-leave_out_samps, :]

    num_rirs, ir_len, n_bands = rirs.shape
    n_slopes = t_vals.shape[1]
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)

    # find the exponential decay envelope for each slope
    # the decay kernel sum_{k=1}^K exp(-t/tau_k) of size n_rir x ir_len x n_slopes x n_bands
    envelopes = np.zeros((num_rirs, ir_len, n_slopes, n_bands))
    for i_slope in range(n_slopes):
        # envelope is in linear scale, not quadratic, therefore decay rates halve, and T values double
        envelope_t = 2 * np.array(t_vals[:, i_slope, ...])

        # generate decay envelopes from t_vals
        envelopes[:, :, i_slope, :] = decay_kernel(envelope_t,
                                                   time,
                                                   fs,
                                                   normalise_envelope=True,
                                                   add_noise=False)

    est_amps = np.zeros((num_rirs, n_slopes, n_bands), dtype=float)
    error = np.zeros_like(est_amps)

    for i in range(num_rirs):
        for k in range(n_bands):
            cur_rir = rirs[i, :, k]
            # psi_k(t)
            cur_edc = calculate_energy_envelope(cur_rir, fs, smooth_time_ms=50)
            cur_edc = cur_edc.reshape(ir_len, 1)
            # psi_k(t) - psi_k(L)
            cur_envelope = envelopes[i, :, :, k] - envelopes[i, -1, :, k]
            assert cur_envelope.shape == (ir_len, n_slopes)
            cur_amps = np.linalg.pinv(cur_envelope) @ (cur_edc)
            error[i, :,
                  k] = np.linalg.norm(cur_envelope @ cur_amps - cur_edc)**2
            # logger.info(
            #     f'num_rir = {i}, error = {20*np.log10(np.abs(error[i,:,k]))} dB'
            # )
            est_amps[i, :, k] = cur_amps

    return est_amps**2


def octave_filtering(input_signal, fs, f_bands, get_filter=False):
    num_bands = len(f_bands)
    out_bands = np.zeros((*input_signal.shape, num_bands))

    for b_idx in range(num_bands):
        if f_bands[b_idx] == 0:
            f_cutoff = (1 / np.sqrt(1.5)) * f_bands[b_idx + 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), output='zpk')
        elif f_bands[b_idx] == fs / 2:
            f_cutoff = np.sqrt(1.5) * f_bands[b_idx - 1]
            z, p, k = butter(5,
                             f_cutoff / (fs / 2),
                             btype='high',
                             output='zpk')
        else:
            this_band = f_bands[b_idx] * np.array(
                [1 / np.sqrt(1.5), np.sqrt(1.5)])
            z, p, k = butter(5,
                             this_band / (fs // 2),
                             btype='band',
                             output='zpk')

        sos = zpk2sos(z, p, k)

        w, h = sosfreqz(sos, worN=len(input_signal) // 2 + 1)
        # somehow this does not work when the input signal is an impulse
        if get_filter:
            out_bands[..., b_idx] = np.fft.irfft(h)
        else:
            out_bands[..., b_idx] = sosfilt(sos, input_signal)
    return out_bands
