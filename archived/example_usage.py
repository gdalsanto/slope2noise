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

    # detect whether the decay time values are breadband of frequency dependent
    if config_dict.f_bands is None:
        n_bands = 1
    else:
        n_bands = len(config_dict.f_bands)

    # sample energy decay parameters using uniform distribution
    # TODO: ideally t_vals, if frequency dependent, should follow a more realistic distribution
    t_vals = np.random.uniform(
        0.5, 3,
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    a_vals = np.random.uniform(
        10**(-3. / 10), 1,
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    _, rirs = shaped_wgn(t_vals,
                        a_vals,
                        fs=config_dict.fs,
                        ir_len=config_dict.ir_len,
                        f_bands=config_dict.f_bands,
                        )

    # save audio files of rirs (only first channel)
    Path(config_dict.output_dir).mkdir(parents=True, exist_ok=True)
    save_audio(os.path.join(config_dict.output_dir, "rir.wav"), rirs[0, :],
               config_dict.fs)

    # test with Bayesian Decay Analysis
    BDA = bda.BayesianDecayAnalysis(config_dict.n_slopes,
                                    config_dict.fs,
                                    filter_frequencies=config_dict.f_bands)

    if config_dict.f_bands is not None:
        subband_rirs = octave_filtering(rirs, config_dict.fs,
                                        config_dict.f_bands)
    else:
        subband_rirs = rirs[..., np.newaxis]

    # get amplitudes with least squares
    A_ls = calculate_amplitudes_least_squares(
        t_vals,
        config_dict.fs,
        subband_rirs,
        config_dict.f_bands,
        leave_out_ms=50.0,
    )

    # go through the parameters of the first 10 rirs
    for i in range(10):
        edc_param, norm_vals = BDA.estimate_parameters(rirs[i, :])
        # BDA takes the EDC parameters estimates for each band
        T, A, N = edc_param[0], edc_param[1], edc_param[2]
        # amplitudes with least squares

        print(
            f"Estimated T: {np.squeeze(np.round(T, 3))}, A: {np.squeeze(np.round(A, 3))}, A_LS: {np.squeeze(np.round(A_ls[i], 3))}" \
            f"  - Reference T: {np.squeeze(np.round(t_vals[i, 0, :], 3))}, A: {np.squeeze(np.round(a_vals[i, 0, :], 3))}"
        )

    # this is of shape ir_len x n_bands
    filtered_rir = octave_filtering(rirs[0, :], config_dict.fs,
                                    config_dict.f_bands)
    time = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs,
                       config_dict.ir_len)
    for j in range(n_bands):
        plt.figure()
        edf = np.flipud(np.cumsum(np.flipud(filtered_rir[:, j]**2), axis=-1))
        plt.plot(time, db(filtered_rir[:, j]))
        plt.plot(time, db(edf, is_squared=True))
        plt.plot(np.zeros(config_dict.n_slopes),
                 db(a_vals[0, :, j], is_squared=True), 'kx')
        plt.plot(np.zeros(config_dict.n_slopes),
                 db(A_ls[0, :, j], is_squared=True), 'gd')
        plt.title(f'RIR at frequency band = {config_dict.f_bands[j]:.0f} Hz')
        plt.ylim([-80, 5])
        plt.show()


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