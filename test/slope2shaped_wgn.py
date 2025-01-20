import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import DecayFitNet.python.toolbox.BayesianDecayAnalysis as bda

from config.config import Config
from slope2noise.generate import *
from slope2noise.utils import *


def main(config_dict: Config):

    # sample energy decay parameters using uniform distribution
    t_vals = np.array(config_dict.t_vals)
    a_vals = np.array(config_dict.a_vals)
    n_vals = np.array(config_dict.n_vals)
    n_rirs, n_slopes, n_bands = t_vals.shape    
    rirs, _ = shaped_wgn(t_vals,    
                        a_vals,
                        n_vals,
                        fs=config_dict.fs,
                        ir_len=config_dict.ir_len,
                        f_bands=config_dict.f_bands,
                        )
    rirs = np.sum(rirs, axis=-2) # sum over the slopes
    # test with Bayesian Decay Analysis
    BDA = bda.BayesianDecayAnalysis(config_dict.n_slopes,
                                    config_dict.fs,
                                    filter_frequencies=config_dict.f_bands)
    edc_param, norm_vals = BDA.estimate_parameters(np.sum(rirs[0, :], -1))
    T_BDA, A_BDA, N = edc_param[0], edc_param[1], edc_param[2]
    T_BDA = np.reshape(T_BDA, (n_rirs, n_slopes, n_bands))
    A_BDA = np.reshape(A_BDA, (n_rirs, n_slopes, n_bands))

    # estimates amplitudes with least squares
    A_LS = calculate_amplitudes_least_squares(
        t_vals,
        config_dict.fs,
        rirs,  
        config_dict.f_bands,
        leave_out_ms=50.0,
    )

    print(
        f"Estimated \nT_BDA: {np.squeeze(np.round(T_BDA, 3))}, A_BDA: {np.squeeze(np.round(A_BDA, 3))} \nA_LS: {np.squeeze(np.round(A_LS[0], 3))}\n" \
        f"Reference \nT: {np.squeeze(np.round(t_vals[0, :, :], 3))}, \nA: {np.squeeze(np.round(a_vals[0, :, :], 3))}"
    )

    # plot Energy Decay Curves
    target_edc = np.zeros((n_rirs, config_dict.ir_len, n_slopes + 1, n_bands))
    edc = np.zeros((n_rirs, config_dict.ir_len, n_bands))
    for i_band in range(n_bands):
        target_edc[..., i_band] = decay_curve(t_vals[:,:,i_band], 
                             a_vals[:,:,i_band],
                             n_vals[:,i_band],
                             fs=config_dict.fs,
                             ir_len=config_dict.ir_len,
                             add_noise=True)
    
        edc[..., i_band] = schroeder_backward_int(rirs[...,i_band], normalize=False)
    time_axis = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs, config_dict.ir_len)

    plt.figure(figsize=(15, 10))
    for i_band in range(n_bands):
        plt.subplot(3, 3, i_band+1)
        plt.plot(time_axis, 10*np.log10(edc[0,:, i_band]), label='shaped noise')
        plt.plot(time_axis, 10*np.log10(np.sum(target_edc[0, :, :, i_band], -1)), '--', label='target')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (dB)')
        plt.legend()
        plt.title(f'EDC at {config_dict.f_bands[i_band]} Hz')
        plt.ylim([-60, 7])
        plt.xlim([0, 2])
        plt.grid(True)
    plt.savefig('test/output/shaped_wgn.png')

    # TODO: Sum the bands and plot the EDC after filtering again fit the filterbank
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_file",
        default=None,
        help="Configuration file (YAML) \
        (if none provided the default parameters are loaded).",
    )

    args = parser.parse_args()

    if args.config_file:
        # Resolve the relative file path
        file_path = Path(args.config_file).resolve()

        # Read and parse the YAML file
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        config_dict = Config(**config_data)
    else:
        config_dict = Config()

    main(config_dict)