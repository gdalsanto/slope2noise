import argparse
import yaml
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from numpy.typing import NDArray, ArrayLike
from typing import Optional, List, Union
from config.config import Config
from slope2noise.rooms import RoomGeometry
from slope2noise.dataclass import CommonSlopesRIR
from slope2noise.generate import shaped_wgn


def generate_amplitudes_based_on_geometry(room: RoomGeometry,
                                          receiver_locs: NDArray,
                                          source_loc: Union[ArrayLike,
                                                            NDArray],
                                          n_slopes: int,
                                          batch_size: int,
                                          pkl_path: Optional[str] = None,
                                          f_bands: Optional[List] = None,
                                          plot: bool = False) -> NDArray:
    """
    Generate amplitudes of shape batch_size x n_slopes based on the coupled space geometry
    Args:
        room (RoomGeometry): room object containing the geometric details of the coupled space
        receiver_locs (NDArray): array of receiver locations of size batch_size x 3
        source_loc (ArrayLike): array of source positions of size 3, or batch_size x 3
        n_slopes (int): number of slopes (corresponds to number of rooms in coupled space)
        batch_size (int): number of receivers
        pkl_path (optional, str): path to file that contains amplitudes distributions analysed from the ThreeRoomDataset
        plot (bool, False): whether to plot the distribution as a function of room geometry
    ReturnsL:
        NDArray: amplitudes of size n_rirs x n_slopes x n_bands
    """
    # read the mean amplitude distribution stored in a pickle file
    if pkl_path is not None:
        with open(Path(pkl_path).resolve(), 'rb') as f:
            gmm_dict = pickle.load(f)

        mean_amps = gmm_dict['means']
        weights = gmm_dict["weights"]
    else:
        weighted_mean = None

    assert n_slopes == room.num_rooms, "number of desired slopes must match number of rooms in coupled space"
    assert receiver_locs.shape[0] == batch_size

    # if frequency-dependent means are provided
    if 'band_centre_hz' in gmm_dict:
        band_centre_hz = gmm_dict['band_centre_hz']
        if f_bands is not None:
            assert len(f_bands) <= len(band_centre_hz)

            # Compute pairwise distances
            # shape: (len(f_bands), len(band_centre_hz))
            dist = np.abs(
                np.array(f_bands)[:, None] - np.array(band_centre_hz)[None, :])

            # For each f_bands[i], find index of closest band centre
            closest_indices = np.argmin(dist, axis=1)

            # Check closeness
            is_close = np.allclose(f_bands, band_centre_hz[closest_indices])

            assert is_close, \
                f"Some f_bands values do not match any band_centre_hz: {f_bands} vs {band_centre_hz}"

        if source_loc.ndim == 1:
            amplitudes_sampled = np.zeros((batch_size, n_slopes, len(f_bands)))
        else:
            amplitudes_sampled = np.zeros(
                (batch_size * source_loc.shape[0], n_slopes, len(f_bands)))

        for k in range(len(f_bands)):
            cur_weighted_mean = weights[closest_indices[k]] * mean_amps[
                closest_indices[k]]
            if source_loc.ndim == 1:
                amplitudes_sampled[
                    ..., k] = room.get_amplitude_based_on_position(
                        receiver_locs, source_loc,
                        cur_weighted_mean[:n_slopes, :n_slopes]).T

                if plot:
                    room.plot_amps_at_receiver_points(
                        receiver_locs,
                        source_loc,
                        amplitudes_sampled[-1],
                        scatter_plot=True,
                        cur_freq_hz=band_centre_hz[closest_indices[k]])
            else:
                for j in range(source_loc.shape[0]):
                    amplitudes_sampled[
                        j * batch_size:(j + 1) * batch_size, :,
                        k] = room.get_amplitude_based_on_position(
                            receiver_locs, source_loc[j],
                            cur_weighted_mean[:n_slopes, :n_slopes]).T

        return amplitudes_sampled

    else:
        weighted_mean = weights * mean_amps
        assert n_slopes <= len(
            mean_amps
        ), "number of desired slopes is greater than the number of slopes in the amplitude distribution"
        if source_loc.ndim == 1:
            # get amplitudes at the specified receiver locations for a single source
            amplitudes_sampled = room.get_amplitude_based_on_position(
                receiver_locs, source_loc,
                weighted_mean[:n_slopes, :n_slopes]).T
            if plot:
                room.plot_amps_at_receiver_points(receiver_locs,
                                                  source_loc,
                                                  amplitudes_sampled,
                                                  scatter_plot=True)
        else:
            # get amplitudes at the receiver locations for multiple sources
            amplitudes_sampled = np.zeros(
                (batch_size * source_loc.shape[0], n_slopes))
            for j in range(source_loc.shape[0]):
                amplitudes_sampled[
                    j * batch_size:(j + 1) *
                    batch_size, :] = room.get_amplitude_based_on_position(
                        receiver_locs, source_loc[j],
                        weighted_mean[:n_slopes, :n_slopes]).T

        return amplitudes_sampled


def gen_dataset(config_dict: Config):

    # get number of batches
    denom = config_dict.batch_size**2 if config_dict.use_multiple_sources else config_dict.batch_size
    if config_dict.n_rirs % denom != 0:
        raise ValueError(
            "Number of RIRs should divide by batch_size if using a single source, or by the square \
            of the batch_size if using many sources")

    n_batch = config_dict.n_rirs // denom
    logger.info(f"Number of batches = {n_batch}")
    num_rooms = config_dict.room_geom_config.num_rooms
    room_dims = config_dict.room_geom_config.room_dims
    start_coordinates = config_dict.room_geom_config.start_coordinates
    aperture_coords = config_dict.room_geom_config.aperture_coords
    t_vals = config_dict.t_vals

    room = RoomGeometry(config_dict.fs, num_rooms, np.array(room_dims),
                        np.array(start_coordinates),
                        config_dict.room_geom_config.aperture_coords)

    for i_batch in range(n_batch):
        # generate source and receiver points
        if config_dict.use_multiple_sources:
            source_loc = room.sample_interior_points(
                n_points=config_dict.batch_size)
        elif config_dict.room_geom_config.source_pos is None:
            source_loc = room.sample_interior_points(n_points=1)
        else:
            source_loc = np.array(config_dict.room_geom_config.source_pos)
        # sample the receiver location within the room
        receiver_locs = room.sample_interior_points(
            n_points=config_dict.batch_size)

        # number of source and receiver points
        num_src_rec_pts = receiver_locs.shape[
            0] if source_loc.ndim == 1 else receiver_locs.shape[
                0] * source_loc.shape[0]

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
                pkl_path=config_dict.amp_gen_config.amp_dist_filepath,
                f_bands=config_dict.f_bands)

        else:
            if config_dict.f_bands is None:
                a_vals = np.random.uniform(
                    10**(-3 / 10), 10**(0 / 10),
                    (num_src_rec_pts, config_dict.n_slopes))
            else:
                a_vals = np.random.uniform(
                    10**(-3 / 10), 10**(0 / 10),
                    (num_src_rec_pts, config_dict.n_slopes,
                     len(config_dict.f_bands)))

        # generate the shaped wgn give the slopes and the decay times
        # t_vals of size batch_size x n_slopes X n_bands
        t_vals_expanded = np.repeat(np.array(t_vals)[np.newaxis, ...],
                                    num_src_rec_pts,
                                    axis=0).squeeze()
        if config_dict.f_bands is None:
            t_vals_expanded = t_vals_expanded[..., 0]

        # synthesize RIRs using the shaped WGN generator
        # shaped_wgn returns (rirs, summed_rirs). rirs has shape
        # (n_rirs, ir_len, n_slopes+1, n_bands)
        rirs, _ = shaped_wgn(
            t_vals_expanded,
            a_vals,
            config_dict.fs,
            config_dict.ir_len,
            f_bands=config_dict.f_bands,
            num_fractions=getattr(config_dict, 'num_fractions', 1),
            filter_length=getattr(config_dict, 'filter_length', 4096),
            compensate_filter_energy=getattr(config_dict,
                                             'compensate_filter_energy', True),
            use_amp_preserving_filterbank=getattr(
                config_dict, 'use_amp_preserving_filterbank', True),
            verbose=False,
        )

        if source_loc.ndim > 1:
            # reshape RIRs and amplitudes to (n_sources, n_receivers, ...)
            rirs = rirs.reshape((source_loc.shape[0], receiver_locs.shape[0]) +
                                rirs.shape[1:])
            new_shape = (source_loc.shape[0],
                         receiver_locs.shape[0]) + a_vals.shape[1:]
            a_vals = a_vals.reshape(new_shape)

        # create instance of CommonSlopes dataclass
        RIRs = CommonSlopesRIR(
            room_dims=room_dims,
            room_start_coords=start_coordinates,
            source_locs=source_loc,
            receiver_locs=receiver_locs,
            n_slopes=config_dict.n_slopes,
            a_vals=a_vals,
            t_vals=t_vals,
            rir=rirs,
            sample_rate=config_dict.fs,
            aperture_coords=config_dict.room_geom_config.aperture_coords,
            f_bands=config_dict.f_bands,
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