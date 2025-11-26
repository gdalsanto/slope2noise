import numpy as np
import argparse
import yaml
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from config.config import Config
from slope2noise.rooms import RoomGeometry
from slope2noise.dataclass import Slope2NoiseUnpickler
from slope2noise.utils import db, calculate_amplitudes_least_squares


def main(config_dict: Config):

    n_rirs = config_dict.n_rirs
    batch_size = config_dict.batch_size
    denom = batch_size**2 if config_dict.use_multiple_sources else batch_size
    num_batches = n_rirs // denom
    n_slopes = config_dict.n_slopes
    n_bands = len(
        config_dict.f_bands) if config_dict.f_bands is not None else 1

    num_receivers = int(
        n_rirs / batch_size) if config_dict.use_multiple_sources else n_rirs
    num_sources = int(n_rirs /
                      batch_size) if config_dict.use_multiple_sources else 1
    receiver_locs = np.zeros((num_receivers, 3))
    source_locs = np.zeros((num_sources, 3))
    a_vals = np.zeros((num_sources, num_receivers, n_slopes, n_bands))
    rirs = np.zeros((num_sources, num_receivers, config_dict.ir_len,
                     n_slopes + 1, n_bands))

    for batch_id in range(num_batches):
        data_path = Path(
            f'{config_dict.output_dir}/bb_wgn_{batch_id:04}.pkl').resolve()
        batch_idx_slice = np.arange(batch_id * batch_size,
                                    (batch_id + 1) * batch_size)
        with open(data_path, 'rb') as f:
            rir_data = Slope2NoiseUnpickler(f).load()

        receiver_locs[batch_idx_slice, :] = rir_data.receiver_locs
        t_vals = np.tile(np.asarray(rir_data.t_vals), (num_receivers, 1, 1))

        if config_dict.use_multiple_sources:
            source_locs[batch_idx_slice, :] = rir_data.source_locs
            a_vals[batch_idx_slice[:, None], batch_idx_slice,
                   ...] = rir_data.a_vals
            rirs[batch_idx_slice[:, None], batch_idx_slice, ...] = rir_data.rir
        else:
            source_locs = rir_data.source_locs[np.newaxis, :]
            a_vals[0, batch_idx_slice, ...] = rir_data.a_vals
            rirs[0, batch_idx_slice, ...] = rir_data.rir

    # plot RIR EDF
    num_rirs_to_plot = 10
    # sum along slopes
    rirs_summed = np.sum(rirs, axis=-2)
    rir_idx = np.random.randint(0, rirs_summed.shape[1], size=num_rirs_to_plot)
    src_idx = 0
    a_vals_est = calculate_amplitudes_least_squares(
        t_vals, config_dict.fs,
        np.expand_dims(rirs_summed[src_idx, ...], axis=-1)
        if config_dict.f_bands is None else rirs_summed[src_idx, ...])

    for k in range(num_rirs_to_plot):
        fig = plt.figure(figsize=(8,
                                  2 * n_bands))  # height scales with n_bands

        for i in range(n_bands):
            ax = fig.add_subplot(n_bands, 1,
                                 i + 1)  # nrows=n_bands, ncols=1, index=i+1

            edf = np.flipud(
                np.cumsum(np.flipud(rirs_summed[src_idx, rir_idx[k], :, i]**2),
                          axis=-1))
            time = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs,
                               config_dict.ir_len)
            ax.plot(time, db(rirs_summed[src_idx, rir_idx[k], :, i]))
            ax.plot(time, db(edf, is_squared=True))
            ax.plot(np.zeros(n_slopes),
                    db(a_vals[src_idx, rir_idx[k], :, i], is_squared=True),
                    'kx',
                    label='a_vals')
            ax.plot(np.zeros(n_slopes),
                    db(a_vals_est[rir_idx[k], :n_slopes, i], is_squared=True),
                    'gd',
                    label='a_vals_est')
            ax.set_xlabel(' Time(s)')
            ax.set_ylabel(f'dB, band = {i+1}')
            if i == 0:
                ax.set_title(
                    f'RIR at receiver pos {receiver_locs[rir_idx[k], 0]:.2f}, {receiver_locs[rir_idx[k], 1]:.2f}, {receiver_locs[rir_idx[k], 2]:.2f}m '
                    +
                    f'at source pos {source_locs[0, 0]:.2f}, {source_locs[0, 1]:.2f}, {source_locs[0, 2]:.2f}m'
                )

        plt.tight_layout()
        plt.show()

        # plot amplitudes as a function of receiver and geometry
        geom_config = config_dict.room_geom_config
        room = RoomGeometry(config_dict.fs, geom_config.num_rooms,
                            geom_config.room_dims,
                            geom_config.start_coordinates,
                            geom_config.aperture_coords)

        for i in range(n_bands):
            if config_dict.use_multiple_sources:
                for n in range(source_locs.shape[0]):
                    room.plot_amps_at_receiver_points(
                        receiver_locs,
                        source_locs[n],
                        a_vals[n, ..., i].T,
                        scatter_plot=False,
                        save_path=Path(
                            f'figures/rir_synthesis_coupled_rooms_amps_source={np.round(source_locs[n], 2)}_freq={config_dict.f_bands[i]:.0f}Hz.png'
                        ).resolve())
            else:
                room.plot_amps_at_receiver_points(
                    receiver_locs,
                    geom_config.source_pos,
                    a_vals[..., i].T,
                    scatter_plot=False,
                    save_path=Path(
                        'figures/rir_synthesis_coupled_rooms_amps_freq={config_dict.f_bands[i]:.0f}Hz.png'
                    ).resolve())


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

    # make output directory if it does not exist
    Path('figures/').mkdir(parents=True, exist_ok=True)
    main(config_dict)