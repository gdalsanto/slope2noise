import numpy as np
from scipy.signal import butter, zpk2sos, sosfreqz, sosfilt
import soundfile as sf

def save_audio(filepath, x, fs=48000):
    sf.write(filepath, x, fs)

def decay_kernel(envelope_t, time, add_noise=True):

    tau_vals = np.log(10**6) / envelope_t
    exponential = np.exp(-np.einsum('nb,t->nbt', tau_vals, time))

    # calculate noise 
    ir_len = len(time)
    noise = np.linspace(1, 1/ir_len, ir_len)

    # construct the decay kernel
    if add_noise:
        return np.concatenate((exponential, noise), axis=0)
    else:
        return exponential

def discard_trailing_zeros(rir):
    # find first non-zero element from back
    last_above_thres = rir.shape[-1] - np.argmax((np.flip(rir,axis=-1) != 0)).squeeze().astype(int)
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
    out = np.cumsum(out ** 2, axis=-1)
    out = np.flip(out, axis=-1)

    # Normalize to 1
    norm_vals = np.max(out, dim=-1, keepdim=True)  # per channel
    out = out / norm_vals

    return out, norm_vals

def octave_filtering(input_signal, fs, f_bands, get_filter=False):
    num_bands = len(f_bands)
    out_bands = np.zeros((*input_signal.shape, num_bands))

    for b_idx in range(num_bands):
        if f_bands[b_idx] == 0:
            f_cutoff = (1 / np.sqrt(1.5)) * f_bands[b_idx + 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), output='zpk')
        elif f_bands[b_idx] == fs / 2:
            f_cutoff = np.sqrt(1.5) * f_bands[b_idx - 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), btype='high', output='zpk')
        else:
            this_band = f_bands[b_idx] * np.array([1 / np.sqrt(1.5), np.sqrt(1.5)])
            z, p, k = butter(5, this_band / (fs // 2), btype='band', output='zpk')

        sos = zpk2sos(z, p, k)

        w, h = sosfreqz(sos, worN = len(input_signal) // 2 + 1)
        # somehow this does not work when the input signal is an impulse 
        if get_filter:
            out_bands[..., b_idx] = np.fft.irfft(h)
        else:
            out_bands[..., b_idx] = sosfilt(sos, input_signal)
    return out_bands