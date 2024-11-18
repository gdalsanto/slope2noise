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

    a_vals = np.random.uniform(
        10**(-3. / 10), 1,
        (config_dict.n_rirs, config_dict.n_slopes, 1))

    _, rirs = shaped_wgn(t_vals,
                        a_vals,
                        fs=config_dict.fs,
                        ir_len=config_dict.ir_len,
                        f_bands=config_dict.f_bands,
                        )

    # test with Bayesian Decay Analysis
    BDA = bda.BayesianDecayAnalysis(config_dict.n_slopes,
                                    config_dict.fs,
                                    filter_frequencies=config_dict.f_bands)
    edc_param, norm_vals = BDA.estimate_parameters(rirs[0, :])
    T_BDA, A_BDA, N = edc_param[0], edc_param[1], edc_param[2]

    # estimates amplitudes with least squares
    A_LS = calculate_amplitudes_least_squares(
        t_vals,
        config_dict.fs,
        rirs[..., np.newaxis],
        config_dict.f_bands,
        leave_out_ms=50.0,
    )

    print(
        f"Estimated \nT_BDA: {np.squeeze(np.round(T_BDA, 3))}, A_BDA: {np.squeeze(np.round(A_BDA, 3))} \nA_LS: {np.squeeze(np.round(A_LS[0], 3))}\n" \
        f"Reference \nT: {np.squeeze(np.round(t_vals[0, :, :], 3))}, A: {np.squeeze(np.round(a_vals[0, :, :], 3))}"
    )

    # plot Energy Decay Curves
    target_edc = decay_curve(t_vals[:,:,0], 
                      a_vals[:,:,0], 
                      fs=config_dict.fs,
                      ir_len=config_dict.ir_len,)
    
    edc = schroeder_backward_int(rirs, normalize=False)
    time_axis = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs, config_dict.ir_len)

    plt.plot(time_axis, 10*np.log10(edc[0, :]),label='generated EDC')
    plt.plot(time_axis, 10*np.log10(np.sum(target_edc[0, :], -1)), '--', label='target EDC')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy Decay')
    plt.legend()
    plt.title(f'Gaussian noise shaping')
    plt.ylim([-40, 5])
    plt.xlim([0, 2])
    plt.grid(True)
    plt.savefig(os.path.join('test/output', 'shaped_wgn_broadband.png'))

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