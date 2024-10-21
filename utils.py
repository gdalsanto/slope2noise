import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Union
from scipy.signal import butter, zpk2sos, sosfreqz, sosfilt
import soundfile as sf


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


def schroeder_backward_int(rir):

    out = discard_trailing_zeros(rir)

    # Backwards integral
    out = np.flip(out, axis=-1)
    out = np.cumsum(out**2, axis=-1)
    out = np.flip(out, axis=-1)

    # Normalize to 1
    norm_vals = np.max(out, dim=-1, keepdim=True)  # per channel
    out = out / norm_vals

    return out, norm_vals


def decay_kernel(envelope_t: Union[float, ArrayLike],
                 time: ArrayLike,
                 add_noise: bool = True) -> NDArray:
    """
    Decay kernel for the exponential envelope
    Args:
        envelope_t: the T60 values (doubled)
        time (ArrayLike): time vector
        add_noise (bool): whether to add noise to the decay kernel
    Returns:
        NDArray: exp(-t/tau) exponential decay kernel, or exp(-t/tau) + n(t) with noise
    """

    tau_vals = np.log(10**6) / envelope_t
    exponential = np.exp(-np.einsum('nb,t->ntb', tau_vals, time))

    # calculate noise
    ir_len = len(time)
    noise = np.linspace(1, 1 / ir_len, ir_len)

    # construct the decay kernel
    if add_noise:
        return np.concatenate((exponential, noise), axis=0)
    else:
        return exponential


def calculate_amplitudes_least_squares(t_vals: NDArray, fs: float,
                                       rirs: NDArray) -> NDArray:
    """
    Calculate amplitudes (one for each slope) using linear least squares
    Args:
        t_vals (NDArray): the T60s of shape n_rir x n_slopes x n_bands
        fs (float): sampling rate
        rirs (NDArray): RIR matrix of shape n_rir x ir_len x n_slopes x n_bands
    Returns:
        NDArray: estimated amplitudes of shape n_rir x n_slopes x n_bands
    """

    if rirs.ndim == 3:
        rirs = np.reshape(rirs,
                          (1, rirs.shape[0], rirs.shape[1], rirs.shape[2]))
        t_vals = t_vals.reshape(1, t_vals.shape[0], t_vals.shape[1])

    num_rirs, ir_len, n_slopes, n_bands = rirs.shape
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)

    # find the exponential decay envelope for each slope
    # the decay kernel sum_{k=1}^K exp(-t/tau_k) of size n_rir x ir_len x n_slopes x n_bands

    envelopes = np.zeros_like(rirs)
    for i_slope in range(n_slopes):
        # envelope is in linear scale, not quadratic, therefore decay rates halve, and T values double
        envelope_t = 2 * np.array(t_vals[:, i_slope, ...])

        # generate decay envelopes from t_vals
        envelopes[:, :, i_slope, :] = decay_kernel(envelope_t,
                                                   time,
                                                   add_noise=False)

    # sum along number of slopes - size is n_rir x ir_len x n_bands
    net_rirs = np.sum(rirs, axis=-2)
    est_amps = np.zeros((num_rirs, n_slopes, n_bands), dtype=float)

    for i in range(num_rirs):
        for k in range(n_bands):
            cur_rir = net_rirs[i, :, k]
            # psi_k(t)
            cur_edc = schroeder_backward_int(cur_rirs).reshape(ir_len, 1)
            # psi_k(t) - psi_k(L)
            cur_envelope = envelopes[i, :, :, k] - envelopes[i, -1, :, k]
            assert cur_envelope.shape == (ir_len, n_slopes)
            cur_amps = np.linalg.pinv(cur_envelope) @ cur_rir
            est_amps[i, :, k] = cur_amps

    return est_amps


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
