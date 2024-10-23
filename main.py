import argparse
import yaml
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from numpy.typing import NDArray, ArrayLike
from typing import Optional
from config.config import Config
from slope2noise.rooms import RoomGeometry, CommonSlopesRIR
from slope2noise.rir_synthesis import rir_synthesis


def generate_amplitudes_based_on_geometry(room: RoomGeometry,
                                          receiver_locs: NDArray,
                                          source_loc: ArrayLike,
                                          n_slopes: int,
                                          batch_size: int,
                                          pkl_path: Optional[str] = None,
                                          plot: bool = False):
    """
    Generate amplitudes of shape batch_size x n_slopes based on the coupled space geometry
    Args:
        room (RoomGeometry): room object containing the geometric details of the coupled space
        receiver_locs (NDArray): array of receiver locations of size batch_size x 3
        source_loc (ArrayLike): source position of size 3
        n_slopes (int): number of slopes (corresponds to number of rooms in coupled space)
        batch_size (int): number of receivers
        pkl_path (optional, str): path to file that contains amplitudes distributions analysed from the ThreeRoomDataset
        plot (bool, False): whether to plot the distribution as a function of room geometry
    """
    # read the mean amplitude distribution stored in a pickle file
    if pkl_path is not None:
        with open(Path(pkl_path).resolve(), 'rb') as f:
            gmm_dict = pickle.load(f)
        mean_amps = gmm_dict['means']
        weights = gmm_dict["weights"]
        assert n_slopes <= len(
            mean_amps
        ), "number of desired slopes is greater than the number of slopes in the amplitude distribution"
        weighted_mean = weights * mean_amps

    else:
        weighted_mean = None

    assert n_slopes == room.num_rooms, "number of desired slopes must match number of rooms in coupled space"
    assert receiver_locs.shape[0] == batch_size

    # get amplitudes at the specified receiver locations
    amplitudes_sampled = room.get_amplitude_based_on_position(
        receiver_locs, source_loc, weighted_mean[:n_slopes, :n_slopes])
    if plot:
        room.plot_amps_at_receiver_points(receiver_locs,
                                          source_loc,
                                          amplitudes_sampled,
                                          scatter_plot=True)

    return amplitudes_sampled.T


def gen_dataset(config_dict: Config):

    # get number of batches
    n_batch = config_dict.n_rirs // config_dict.batch_size
    if config_dict.n_rirs % config_dict.batch_size != 0:
        n_batch += 1

    num_rooms = config_dict.room_geom_config.num_rooms
    room_dims = config_dict.room_geom_config.room_dims
    start_coordinates = config_dict.room_geom_config.start_coordinates
    t_vals = config_dict.t_vals

    room = RoomGeometry(config_dict.fs, num_rooms, np.array(room_dims),
                        np.array(start_coordinates))

    for i_batch in range(n_batch):
        if config_dict.room_geom_config is None:
            source_loc = room.sample_interior_points(n_points=1)
        else:
            source_loc = np.array(config_dict.room_geom_config.source_pos)
        # sample the receiver location within the room
        receiver_locs = room.sample_interior_points(
            n_points=config_dict.batch_size)

        # get the amplitudes of the slopes
        # generate broadband amplitudes based on the coupled space geometry
        if config_dict.amp_gen_config.generate_amplitude_based_on_geometry:
            a_vals = generate_amplitudes_based_on_geometry(
                room,
                receiver_locs,
                source_loc,
                config_dict.n_slopes,
                config_dict.batch_size,
                plot=False,
                pkl_path=config_dict.amp_gen_config.amp_dist_filepath)

        else:
            a_vals = np.random.uniform(
                10**(-3 / 10), 10**(0 / 10),
                (config_dict.batch_size, config_dict.n_slopes))
        # generate the shaped wgn give the slopes and the decay times
        # t_vals of size batch_size x n_slopes
        _, rirs = rir_synthesis(np.repeat(np.array(t_vals)[np.newaxis, :],
                                          config_dict.batch_size,
                                          axis=0),
                                a_vals,
                                config_dict.f_bands,
                                config_dict.fs,
                                config_dict.ir_len,
                                type=config_dict.synthesis_type)
        # create instance of CommonSlopes dataclass
        RIRs = CommonSlopesRIR(source_locs=source_loc,
                               receiver_locs=receiver_locs,
                               n_slopes=config_dict.n_slopes,
                               a_vals=a_vals,
                               t_vals=t_vals,
                               rir=rirs,
                               batch_id=i_batch)
        # save it to a pkl file
        with open(
                os.path.join(config_dict.output_dir,
                             f"bb_wgn_{i_batch:04}.pkl"), "wb") as f:
            pickle.dump(RIRs, f)


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
    Path(config_dict.output_dir).mkdir(parents=True, exist_ok=True)

    gen_dataset(config_dict)