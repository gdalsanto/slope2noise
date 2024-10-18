import numpy as np 
from utils import * 
import matplotlib.pyplot as plt

def decay_kernel(envelope_t, time, add_noise=True):

    tau_vals = np.log(10**6) / envelope_t
    exponential = np.exp(-np.einsum('nb,t->ntb', tau_vals, time))

    # calculate noise 
    ir_len = len(time)
    noise = np.linspace(1, 1/ir_len, ir_len)

    # construct the decay kernel
    if add_noise:
        return np.concatenate((exponential, noise), axis=0)
    else:
        return exponential


def modal_synthesis(t_vals, a_vals, f_bands, n_modes, fs, ir_len, type='modal'):

    # freuqencey bands are assumend to be log spaced

    # assert input dimensions
    assert len(t_vals.shape) >= 1 and len(t_vals.shape) <= 3, 'Incorrect dimension for t_vals. Must be either [n_rir x n_slopes x n_bands] or [n_rir x n_slopes] or [n_rir].'
    assert len(a_vals.shape) == len(t_vals.shape) <= 3, 'Incorrect dimension for a_vals. Must be the same as t_vals.'
    
    # expand dimensions if necessary
    if len(t_vals.shape) < 3:
        t_vals = np.reshape(t_vals, (*t_vals.shape,  *tuple([1 for d in range(3-len(t_vals.shape))])))
        a_vals = np.reshape(a_vals, (*a_vals.shape,  *tuple([1 for d in range(3-len(a_vals.shape))])))
    if f_bands is not None:
        assert t_vals.shape[-1] == len(f_bands), 'Mismatch in number of bands. t_vals should have appropriate dimensions.'
        if type == 'modal':
            # modal synthesis requires side bands for interpolation
            n_bands = len(f_bands) + 2 
        else:
            n_bands = len(f_bands)
    else:
        n_bands = 1 # broadband

    n_rirs = t_vals.shape[0]
    n_slopes = t_vals.shape[1]

    # time arrray 
    time = np.linspace(0, (ir_len - 1) / fs, ir_len)

    # initialize output arrays
    gaussian_noise = np.zeros((n_rirs, ir_len, n_slopes, n_bands))  # without envelopes
    envelope_a = np.zeros((n_rirs, ir_len, n_slopes, n_bands)) 

    for i_slope in range(n_slopes):
        
        # envelope is in linear scale, not quadratic, therefore decay rates halve, and T values double
        envelope_t = 2 * np.array(t_vals[:, i_slope, ...])

        # generate decay envelopes from t_vals
        envelopes = decay_kernel(envelope_t, time, add_noise=False)
        
        if n_bands > 1 and type == 'modal':
            # duplicate first and last envelope for residual band
            envelopes = np.concatenate([np.expand_dims(envelopes[:, :, 0], -1), envelopes, np.expand_dims(envelopes[:, :, -1], -1)], axis=-1)  
            a_vals = np.concatenate([np.expand_dims(a_vals[:, :, 0], 1), a_vals, np.expand_dims(a_vals[:, :, -1], 1)], axis=-1)  

        # convert envelope to dB
        envelopes_db = 10 * np.log10(envelopes) 

        # output arrays
        synthesis_rirs = np.zeros((n_rirs, ir_len, n_slopes, n_bands)) 
        if type == 'modal':

            # random mode phase
            mode_phase = 2 * np.pi * np.random.rand(n_rirs, n_modes)
            # randomly draw log-spaced frequencies
            mode_freq = np.expand_dims(np.logspace(np.log10(1), np.log10(0.99 * fs / 2), n_modes), axis=0)

            # # add jitter TODO fix jitter (as of now cannot be done this way for n_rirs > 1)
            # jitter = np.mean(np.diff(np.log10(mode_freq[0,:]), axis=-1)) 
            # mode_freq = 10 ** (np.log10(mode_freq) + np.random.rand(n_rirs, n_modes) * jitter)
            # remove mode frequencies close to aliasing
            # mode_freq = mode_freq[mode_freq < 0.99 * fs / 2]
            # n_modes = len(mode_freq)

            # for mode frequencies between band centers: interpolate position in log
            # scale and combine their bands' contributions. Decay rate of
            # "in-between" modes should be in between the center frequency rates
            band_fraction = np.interp(np.log10(mode_freq), np.log10([1, *f_bands, fs / 2]), np.arange(1, n_bands+1)).squeeze()
            i_band = np.round(band_fraction).astype(int)-1
            band_floor = np.floor(band_fraction).astype(int)
            band_mix = band_fraction - band_floor

            for i_mode in range(n_modes):
                # Sinusoidal mode with specified frequency and phase
                # TODO add aption either mode or wgn 
                mode = np.sin(2 * np.pi * np.einsum('n, t -> nt', mode_freq[:, i_mode], time) + np.expand_dims(mode_phase[:, i_mode], axis=-1))

                gaussian_noise[:, :, i_slope, i_band[i_mode]] += mode

                # Interpolate envelope in dB scale
                inter_envelope_db = (1 - band_mix[i_mode]) * envelopes_db[:, :, band_floor[i_mode]-1] + band_mix[i_mode] * envelopes_db[:, :, band_floor[i_mode]]
                interp_envelope = 10 ** (inter_envelope_db / 10)  # Convert back to linear scale

                # Apply envelope to the sinusoidal mode
                mode = mode * interp_envelope

                # Add the mode to the corresponding band
                synthesis_rirs[:, :, i_slope, i_band[i_mode]] += mode

                # Save amplitude envelopes
                envelope_a[:, :, i_slope, i_band[i_mode]] = np.expand_dims(np.sqrt(a_vals[:, i_slope, i_band[i_mode]]), axis=-1) * np.sqrt((1-interp_envelope))
        
        elif type == 'wgn':
            
            # generate random sequence of Gaussian noise
            random_sequence = np.random.randn(n_rirs, ir_len, n_slopes, 1)


            filtered_noise = np.zeros((n_rirs, ir_len, n_slopes, n_bands))

            if n_bands > 1:
                # get energy of the filter bank
                inpulse = np.zeros((ir_len))
                inpulse[0] = 1
                ir_octave_filter = octave_filtering(inpulse, fs, f_bands, get_filter=True)  # the input inpulse will not be used in this case actually, get_filter argument is just a quick fix
                band_energy = sum(ir_octave_filter**2,0)
                # fitler the random sequence in frequency to extract the band
                filtered_noise[:, :, i_slope, :] = octave_filtering(random_sequence[:, :, i_slope, 0], fs, f_bands)
                for i_band in range(n_bands):
                    gaussian_noise[:, :, i_slope, i_band] = np.einsum('nt, nt -> nt', filtered_noise[:, :, i_slope, i_band], envelopes[..., i_band])
                    envelope_a[:, :, i_slope, i_band] = np.expand_dims(np.sqrt(a_vals[:, i_slope, i_band]), axis=-1) * np.sqrt((1-envelopes[..., i_band]) / band_energy[i_band])
            else:
                gaussian_noise[:, :, i_slope, :] = np.einsum('ntb, ntb -> ntb', random_sequence[:, :, i_slope, :], envelopes)
                envelope_a[:, :, i_slope, 0] = np.sqrt(a_vals[:, i_slope, :]) * np.sqrt((1-envelopes[:, :, 0]) )

    # Scale to ensure unit RMS for each band
    scaling_factor = 1.0 / np.sqrt(np.mean(np.square(gaussian_noise), axis=1))
    synthesis_rirs = gaussian_noise * np.expand_dims(scaling_factor, axis=1) 
    # Apply amplitude envelopes
    synthesis_rirs *= envelope_a
    return synthesis_rirs, synthesis_rirs.sum(axis=-1).sum(axis=-1)
