import numpy as np
import argparse
import yaml
import pickle
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from config.config import Config
from slope2noise.rooms import RoomGeometry
from slope2noise.utils import db


def main(config_dict: Config):
    n_rirs = config_dict.n_rirs
    batch_size = config_dict.batch_size
    num_batches = int(n_rirs / batch_size)
    n_slopes = config_dict.n_slopes

    receiver_locs = np.zeros((n_rirs, 3))
    a_vals = np.zeros((n_rirs, n_slopes))
    rirs = np.zeros((n_rirs, config_dict.ir_len))
    for batch_id in range(num_batches):
        data_path = Path(f'output/bb_wgn_{batch_id:04}.pkl').resolve()
        batch_idx_slice = np.arange(batch_id * batch_size,
                                    (batch_id + 1) * batch_size)

        with open(data_path, 'rb') as f:
            rir_data = pickle.load(f)

        receiver_locs[batch_idx_slice, :] = rir_data.receiver_locs
        a_vals[batch_idx_slice, :] = rir_data.a_vals
        rirs[batch_idx_slice, :] = rir_data.rir

    # plot RIR EDF
    num_rirs_to_plot = 10
    rir_idx = np.random.randint(0, n_rirs, size=num_rirs_to_plot)

    for k in range(num_rirs_to_plot):
        plt.figure()
        edf = np.flipud(np.cumsum(np.flipud(rirs[rir_idx[k], :]**2), axis=-1))
        time = np.linspace(0, (config_dict.ir_len - 1) / config_dict.fs,
                           config_dict.ir_len)
        plt.plot(time, db(rirs[rir_idx[k], :]))
        plt.plot(time, db(edf, is_squared=True))
        plt.plot(np.zeros(n_slopes), db(a_vals[rir_idx[k], :],
                                        is_squared=True), 'kx')
        plt.title(
            f'RIR at position {receiver_locs[rir_idx[k], 0]:.2f}, {receiver_locs[rir_idx[k], 1]:.2f}, {receiver_locs[rir_idx[k], 2]:.2f} m'
        )
        plt.show()

    # plot amplitudes as a function of receiver and geometry
    geom_config = config_dict.room_geom_config
    room = RoomGeometry(config_dict.fs, geom_config.num_rooms,
                        geom_config.room_dims, geom_config.start_coordinates)
    room.plot_amps_at_receiver_points(
        receiver_locs,
        geom_config.source_pos,
        a_vals.T,
        scatter_plot=False,
        save_path=Path(
            'figures/rir_synthesis_coupled_rooms_amps.png').resolve())


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