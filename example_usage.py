import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import DecayFitNet.python.toolbox.BayesianDecayAnalysis as bda

from config.config import Config
from slope2noise.rir_synthesis import rir_synthesis
from slope2noise.utils import schroeder_backward_int, save_audio, calculate_amplitudes_least_squares


def main(config_dict: Config):

    # detect whether the decay time values are breadband of frequency dependent
    if config_dict.f_bands is None:
        n_bands = 1
    else:
        n_bands = len(config_dict.f_bands)

    # sample energy decay parameters using uniform distribution
    # TODO: ideally t_vals, if frequency dependent, should follow a more realistic distribution
    t_vals = np.random.uniform(
        config_dict.t_vals_lims[0], config_dict.t_vals_lims[1],
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    a_vals = np.random.uniform(
        config_dict.a_vals_lims[0], config_dict.a_vals_lims[1],
        (config_dict.n_rirs, config_dict.n_slopes, n_bands))

    rirs_per_slope, rirs = rir_synthesis(
        t_vals,
        a_vals,
        config_dict.f_bands,
        config_dict.fs,
        config_dict.ir_len,
        type=config_dict.synthesis_type,
        n_modes=config_dict.n_modes,
    )

    edf = np.flipud(np.cumsum(np.flipud(rirs[0, :]**2), axis=-1))
    time = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs,
                       config_dict.ir_len)
    plt.plot(time, 10 * np.log10(edf))
    plt.plot(0, 10 * np.log10(a_vals[0]), 'kx')
    plt.show()
    plt.savefig("edf.png")

    # save audio files of rirs (only first channel)
    Path(config_dict.output_dir).mkdir(parents=True, exist_ok=True)
    save_audio(os.path.join(config_dict.output_dir, "rir.wav"), rirs[0, :],
               config_dict.fs)
    # test with Bayesian Decay Analysis
    BDA = bda.BayesianDecayAnalysis(config_dict.n_slopes,
                                    config_dict.fs,
                                    filter_frequencies=config_dict.f_bands)

    # get amplitudes with least squares
    A_ls = calculate_amplitudes_least_squares(t_vals,
                                              config_dict.fs,
                                              rirs[..., np.newaxis],
                                              leave_out_ms=50.0)

    # go through the parameters of the first 10 rirs
    for i in range(10):
        edc_param, norm_vals = BDA.estimate_parameters(rirs[i, :])
        # BDA takes the EDC parameters estimates for each band
        T, A, N = np.mean(edc_param[0]), np.mean(edc_param[1]), np.mean(
            edc_param[2])
        # amplitudes with least squares

        print(
            f"Estimated T: {T:.3f}, A: {A:.3f}, A_LS: {np.squeeze(A_ls[i]):.3f}  - Reference T: {t_vals[i, 0, 0]:.3f}, A: {a_vals[i, 0, 0]:.3f}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_file",
        default=None,
        help="Configuration file (YAML) containing diff GFDN \
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